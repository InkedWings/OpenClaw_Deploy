#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   # inside an active salloc allocation
#   export WORK=/lustre/orion/gen150/scratch/zye25/Agentic
#   export IMG=$WORK/containers/vllm_rocm631_vllm083.sif
#   bash script/manage_vllm_backends.sh start
#   bash script/manage_vllm_backends.sh status
#   bash script/manage_vllm_backends.sh stop
#
# Optional overrides (env):
#   MODEL, PORT, TP_SIZE, GPU_MEMORY_UTILIZATION
#   GPU_DEVICES or ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES
#   NET_IFACE, VLLM_HOST_IP, GLOO_SOCKET_IFNAME, NCCL_SOCKET_IFNAME, RCCL_SOCKET_IFNAME, GLOO_USE_IPV6
#   VLLM_USE_TRITON_FLASH_ATTN
#   BACKEND_API_KEY (optional; if set, backend requires bearer auth and readiness probes send it)
#   RUN_TAG, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE
#   ENABLE_AUTO_TOOL_CHOICE, TOOL_CALL_PARSER, TOOL_PARSER_PLUGIN, VLLM_EXTRA_ARGS
#   WAIT_BACKENDS_READY, READY_TIMEOUT_S, READY_POLL_INTERVAL_S, READY_CURL_TIMEOUT_S, READY_PROGRESS

ACTION="${1:-start}"
SLURM_TARGET_JOB_ID="${SLURM_TARGET_JOB_ID:-${SLURM_JOB_ID:-}}"

if [[ -z "${SLURM_JOB_NODELIST:-}" && -n "${SLURM_TARGET_JOB_ID}" ]]; then
  SLURM_JOB_NODELIST="$(squeue -h -j "${SLURM_TARGET_JOB_ID}" -o %N | head -n 1)"
fi

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "[error] SLURM_JOB_NODELIST is empty."
  echo "        Run inside an active salloc/sbatch shell, or set SLURM_TARGET_JOB_ID=<jobid>."
  exit 1
fi

WORK="${WORK:-$PWD}"
IMG="${IMG:-$WORK/containers/vllm_rocm631_vllm083.sif}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-${GPU_DEVICES}}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES}}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
TOOL_PARSER_PLUGIN="${TOOL_PARSER_PLUGIN:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
NET_IFACE="${NET_IFACE:-hsn0}"
VLLM_HOST_IP="${VLLM_HOST_IP:-}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
RCCL_SOCKET_IFNAME="${RCCL_SOCKET_IFNAME:-}"
GLOO_USE_IPV6="${GLOO_USE_IPV6:-0}"
# On ROCm + Qwen2.x (SWA), Triton FA can be slow/unstable; default to CK FA.
# You can still override per-run by exporting VLLM_USE_TRITON_FLASH_ATTN=1.
VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"
BACKEND_API_KEY="${BACKEND_API_KEY:-}"
WAIT_BACKENDS_READY="${WAIT_BACKENDS_READY:-1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
READY_POLL_INTERVAL_S="${READY_POLL_INTERVAL_S:-30}"
READY_CURL_TIMEOUT_S="${READY_CURL_TIMEOUT_S:-4}"
# Print per-poll readiness progress (1=on, 0=off).
READY_PROGRESS="${READY_PROGRESS:-1}"

VLLM_TOOL_ARGS=""
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  VLLM_TOOL_ARGS="--enable-auto-tool-choice --tool-call-parser ${TOOL_CALL_PARSER}"
  if [[ -n "${TOOL_PARSER_PLUGIN}" ]]; then
    VLLM_TOOL_ARGS+=" --tool-parser-plugin ${TOOL_PARSER_PLUGIN}"
  fi
fi

VLLM_LAUNCH_ARGS="${VLLM_TOOL_ARGS} ${VLLM_EXTRA_ARGS}"

RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}_p${PORT}}"
STATE_DIR="${WORK}/logs/vllm_launcher"
LOG_DIR="${WORK}/logs"
PID_FILE="${STATE_DIR}/${RUN_TAG}.pid"
HOSTS_FILE="${STATE_DIR}/${RUN_TAG}.hosts"
ENDPOINTS_FILE="${STATE_DIR}/${RUN_TAG}.endpoints.sh"
SRUN_LOG="${STATE_DIR}/${RUN_TAG}.srun.log"

mkdir -p "${STATE_DIR}" "${LOG_DIR}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "[error] No hosts found from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
  exit 1
fi

write_endpoints_file() {
  : > "${ENDPOINTS_FILE}"
  local idx=1
  local urls=()
  for host in "${HOSTS[@]}"; do
    local url="http://${host}:${PORT}/v1"
    urls+=("${url}")
    printf 'export ENDPOINT_%d=%q\n' "${idx}" "${url}" >> "${ENDPOINTS_FILE}"
    idx=$((idx + 1))
  done
  if [[ "${#HOSTS[@]}" -ge 1 ]]; then
    printf 'export ENDPOINT_A=%q\n' "http://${HOSTS[0]}:${PORT}/v1" >> "${ENDPOINTS_FILE}"
  fi
  if [[ "${#HOSTS[@]}" -ge 2 ]]; then
    printf 'export ENDPOINT_B=%q\n' "http://${HOSTS[1]}:${PORT}/v1" >> "${ENDPOINTS_FILE}"
  fi
  printf 'export BASE_URLS=%q\n' "$(IFS=,; echo "${urls[*]}")" >> "${ENDPOINTS_FILE}"
}

print_endpoints() {
  local idx=1
  for host in "${HOSTS[@]}"; do
    echo "ENDPOINT_${idx}=http://${host}:${PORT}/v1"
    idx=$((idx + 1))
  done
}

endpoint_ready() {
  local url="$1"
  local body
  if [[ -n "${BACKEND_API_KEY}" ]]; then
    body="$(curl -fsS --max-time "${READY_CURL_TIMEOUT_S}" -H "Authorization: Bearer ${BACKEND_API_KEY}" "${url}" 2>/dev/null || true)"
  else
    body="$(curl -fsS --max-time "${READY_CURL_TIMEOUT_S}" "${url}" 2>/dev/null || true)"
  fi
  [[ -n "${body}" && "${body}" == *'"data"'* ]]
}

wait_backends_ready() {
  if [[ "${WAIT_BACKENDS_READY}" != "1" ]]; then
    echo "[info] Skipping readiness wait (WAIT_BACKENDS_READY=${WAIT_BACKENDS_READY})."
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "[warn] curl not found. Skip readiness wait."
    return 0
  fi

  echo "[info] Waiting for vLLM endpoints to be ready..."
  echo "[info] timeout=${READY_TIMEOUT_S}s poll=${READY_POLL_INTERVAL_S}s curl_timeout=${READY_CURL_TIMEOUT_S}s"

  local deadline now idx all_ready host url start_ts elapsed ready_count
  deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
  start_ts=$(date +%s)
  local -a ready
  ready=()
  for _ in "${HOSTS[@]}"; do
    ready+=("0")
  done

  while true; do
    all_ready=1
    ready_count=0
    local -a pending_urls
    pending_urls=()
    idx=0
    for host in "${HOSTS[@]}"; do
      url="http://${host}:${PORT}/v1/models"
      if [[ "${ready[${idx}]}" == "1" ]]; then
        ready_count=$((ready_count + 1))
        idx=$((idx + 1))
        continue
      fi

      if endpoint_ready "${url}"; then
        ready[${idx}]="1"
        ready_count=$((ready_count + 1))
        echo "[ready] ${url}"
      else
        all_ready=0
        pending_urls+=("${url}")
      fi
      idx=$((idx + 1))
    done

    if [[ "${all_ready}" == "1" ]]; then
      echo "[ok] All endpoints are ready."
      return 0
    fi

    now=$(date +%s)
    if [[ "${READY_PROGRESS}" == "1" ]]; then
      elapsed=$((now - start_ts))
      echo "[wait] elapsed=${elapsed}s ready=${ready_count}/${#HOSTS[@]} pending=${pending_urls[*]}"
    fi
    if (( now >= deadline )); then
      echo "[warn] Timed out waiting for readiness after ${READY_TIMEOUT_S}s."
      idx=0
      for host in "${HOSTS[@]}"; do
        if [[ "${ready[${idx}]}" != "1" ]]; then
          echo "[warn] not ready: http://${host}:${PORT}/v1/models"
        fi
        idx=$((idx + 1))
      done
      echo "[hint] Check per-node logs: ${LOG_DIR}/vllm_<hostname>_${PORT}.log"
      return 1
    fi

    sleep "${READY_POLL_INTERVAL_S}"
  done
}

start_backends() {
  if [[ -f "${PID_FILE}" ]]; then
    local old_pid
    old_pid="$(cat "${PID_FILE}")"
    if ps -p "${old_pid}" >/dev/null 2>&1; then
      echo "[info] Existing launcher is still running (pid=${old_pid}, run_tag=${RUN_TAG})."
      echo "[info] Use: bash script/manage_vllm_backends.sh status"
      return 0
    fi
    rm -f "${PID_FILE}"
  fi

  printf '%s\n' "${HOSTS[@]}" > "${HOSTS_FILE}"
  write_endpoints_file

  echo "[info] Starting vLLM on ${#HOSTS[@]} nodes with run_tag=${RUN_TAG}"
  print_endpoints
  echo "[info] Per-node logs: ${LOG_DIR}/vllm_<hostname>_${PORT}.log"
  echo "[info] GPU masks: ROCR=${ROCR_VISIBLE_DEVICES} HIP=${HIP_VISIBLE_DEVICES} CUDA=${CUDA_VISIBLE_DEVICES}"
  echo "[info] Net defaults: NET_IFACE=${NET_IFACE} VLLM_HOST_IP=${VLLM_HOST_IP:-<auto>} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<auto>} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<auto>} VLLM_USE_TRITON_FLASH_ATTN=${VLLM_USE_TRITON_FLASH_ATTN}"
  if [[ -n "${BACKEND_API_KEY}" ]]; then
    echo "[info] Backend auth: enabled (Bearer token required)."
  else
    echo "[info] Backend auth: disabled."
  fi
  if [[ -n "${VLLM_LAUNCH_ARGS// }" ]]; then
    echo "[info] Extra vLLM args: ${VLLM_LAUNCH_ARGS}"
  fi

  local srun_job_arg=()
  if [[ -n "${SLURM_TARGET_JOB_ID}" ]]; then
    srun_job_arg+=(--jobid "${SLURM_TARGET_JOB_ID}")
  fi
  local hostlist
  hostlist="$(IFS=,; echo "${HOSTS[*]}")"

  nohup srun \
    "${srun_job_arg[@]}" \
    -N "${#HOSTS[@]}" \
    -n "${#HOSTS[@]}" \
    --ntasks-per-node=1 \
    --nodelist "${hostlist}" \
    --overlap \
    bash -lc '
      set -euo pipefail
      host="${SLURMD_NODENAME:-$(hostname -s)}"
      net_iface="'"${NET_IFACE}"'"
      vllm_host_ip="'"${VLLM_HOST_IP}"'"
      gloo_if="'"${GLOO_SOCKET_IFNAME}"'"
      nccl_if="'"${NCCL_SOCKET_IFNAME}"'"
      rccl_if="'"${RCCL_SOCKET_IFNAME}"'"
      gloo_use_ipv6="'"${GLOO_USE_IPV6}"'"
      backend_api_key="'"${BACKEND_API_KEY}"'"

      if [[ -z "${vllm_host_ip}" ]]; then
        if ip -o -4 addr show "${net_iface}" >/dev/null 2>&1; then
          vllm_host_ip="$(ip -o -4 addr show "${net_iface}" | awk "{print \$4}" | cut -d/ -f1 | head -n1)"
        fi
        if [[ -z "${vllm_host_ip}" ]]; then
          vllm_host_ip="$(hostname -I | awk "{print \$1}")"
        fi
      fi
      if [[ -z "${vllm_host_ip}" ]]; then
        vllm_host_ip="127.0.0.1"
      fi

      if [[ -z "${gloo_if}" ]]; then
        if [[ "${vllm_host_ip}" == 127.* ]]; then
          gloo_if="lo"
        else
          gloo_if="${net_iface}"
        fi
      fi
      if [[ -z "${nccl_if}" ]]; then
        nccl_if="${gloo_if}"
      fi
      if [[ -z "${rccl_if}" ]]; then
        rccl_if="${nccl_if}"
      fi

      mkdir -p "'"${LOG_DIR}"'"
      log="'"${LOG_DIR}"'/vllm_${host}_'"${PORT}"'.log"
      echo "[start] host=${host} port='"${PORT}"' log=${log}"
      echo "[net] VLLM_HOST_IP=${vllm_host_ip} GLOO_SOCKET_IFNAME=${gloo_if} NCCL_SOCKET_IFNAME=${nccl_if} RCCL_SOCKET_IFNAME=${rccl_if} GLOO_USE_IPV6=${gloo_use_ipv6}"

      exec apptainer exec \
        --bind "'"${WORK}"'":/workspace \
        --env HF_HOME=/workspace/hf \
        --env HUGGINGFACE_HUB_CACHE=/workspace/hf \
        --env HF_HUB_OFFLINE="'"${HF_HUB_OFFLINE}"'" \
        --env TRANSFORMERS_OFFLINE="'"${TRANSFORMERS_OFFLINE}"'" \
        --env ROCR_VISIBLE_DEVICES="'"${ROCR_VISIBLE_DEVICES}"'" \
        --env HIP_VISIBLE_DEVICES="'"${HIP_VISIBLE_DEVICES}"'" \
        --env CUDA_VISIBLE_DEVICES="'"${CUDA_VISIBLE_DEVICES}"'" \
        --env VLLM_HOST_IP="${vllm_host_ip}" \
        --env GLOO_SOCKET_IFNAME="${gloo_if}" \
        --env GLOO_USE_IPV6="${gloo_use_ipv6}" \
        --env NCCL_SOCKET_IFNAME="${nccl_if}" \
        --env RCCL_SOCKET_IFNAME="${rccl_if}" \
        --env VLLM_USE_TRITON_FLASH_ATTN="'"${VLLM_USE_TRITON_FLASH_ATTN}"'" \
        --env VLLM_API_KEY= \
        "'"${IMG}"'" \
        python -m vllm.entrypoints.openai.api_server \
          --model "'"${MODEL}"'" \
          --host 0.0.0.0 --port "'"${PORT}"'" \
          --tensor-parallel-size "'"${TP_SIZE}"'" \
          --gpu-memory-utilization "'"${GPU_MEMORY_UTILIZATION}"'" \
          ${backend_api_key:+--api-key "${backend_api_key}"} \
          '"${VLLM_LAUNCH_ARGS}"' \
        > "${log}" 2>&1
    ' > "${SRUN_LOG}" 2>&1 &

  local launcher_pid=$!
  echo "${launcher_pid}" > "${PID_FILE}"

  echo "[ok] Launcher started. pid=${launcher_pid}"
  echo "[ok] srun log: ${SRUN_LOG}"
  echo "[ok] endpoints env file: ${ENDPOINTS_FILE}"
  echo "[hint] Load endpoints:"
  echo "       source ${ENDPOINTS_FILE}"

  wait_backends_ready
}

stop_backends() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "[info] No pid file found for run_tag=${RUN_TAG}. Nothing to stop."
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  if ps -p "${pid}" >/dev/null 2>&1; then
    echo "[info] Stopping launcher pid=${pid} (run_tag=${RUN_TAG})"
    kill "${pid}" || true
    sleep 1
    if ps -p "${pid}" >/dev/null 2>&1; then
      echo "[warn] pid=${pid} still alive, sending SIGKILL"
      kill -9 "${pid}" || true
    fi
    echo "[ok] Stopped."
  else
    echo "[info] Launcher pid=${pid} is not running."
  fi

  rm -f "${PID_FILE}"
}

status_backends() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      echo "[ok] Launcher running. pid=${pid}"
    else
      echo "[warn] pid file exists but process is not running. pid=${pid}"
    fi
  else
    echo "[info] No launcher pid file for run_tag=${RUN_TAG}"
  fi

  if [[ -f "${ENDPOINTS_FILE}" ]]; then
    echo "[info] endpoints file: ${ENDPOINTS_FILE}"
    sed -n '1,120p' "${ENDPOINTS_FILE}"
  else
    echo "[info] endpoints file not found yet."
    print_endpoints
  fi
}

case "${ACTION}" in
  start)
    start_backends
    ;;
  stop)
    stop_backends
    ;;
  status)
    status_backends
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 2
    ;;
esac
