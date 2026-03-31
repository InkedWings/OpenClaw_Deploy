## 1. Request New Compute Nodes

```bash
salloc -A gen150 -q debug -N 2 -t 02:00:00
squeue -u $USER -o "%.18i %.8T %.30j %R"
```

---

## 2. Start vLLM (Llama 3.1 70B Instruct, TP=8)

```bash
module load olcf-container-tools
module load apptainer-enable-mpi
module load apptainer-enable-gpu

export WORK=/lustre/orion/gen150/scratch/zye25/Agentic
export IMG=$WORK/containers/vllm_rocm631_vllm083.sif
export SLURM_TARGET_JOB_ID=$SLURM_JOB_ID

export MODEL=/workspace/hf/models/Llama-3.1-70B-Instruct
export PORT=8011
export TP_SIZE=4
export GPU_DEVICES=0,1,2,3
export GPU_MEMORY_UTILIZATION=0.90
export VLLM_EXTRA_ARGS='--served-model-name meta-llama/Llama-3.1-70B-Instruct'

# Readiness Wait Policy
export READY_TIMEOUT_S=3600
export READY_POLL_INTERVAL_S=15
export READY_PROGRESS=1

cd $WORK
bash script/manage_vllm_backends.sh stop
bash script/manage_vllm_backends.sh start
```

```bash
source "$(ls -1t $WORK/logs/vllm_launcher/*_p${PORT}.endpoints.sh | head -n1)"
echo "$ENDPOINT_A"
echo "$ENDPOINT_B"
```

---

## 3. Create Isolated Directory

```bash
cd $WORK
mkdir -p openclaw_latest/{bin,logs,state,.local,cache}
cd openclaw_latest
```

---

## 4. Install Local Node 22

### 4.1 Proxy Settings

```bash
export PROXY=http://proxy.ccs.ornl.gov:3128
export http_proxy=$PROXY
export https_proxy=$PROXY
export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY
SLURM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
SLURM_IPS=$(getent hosts $(scontrol show hostnames "$SLURM_JOB_NODELIST") | awk '{print $1}' | paste -sd, -)

export no_proxy="localhost,127.0.0.1,::1,.ornl.gov,.olcf.ornl.gov,.frontier.olcf.ornl.gov,$(hostname -s),$(hostname -f),$SLURM_HOSTS,$SLURM_IPS"
export NO_PROXY="$no_proxy"
```

### 4.2 Download and Install Node

```bash
cd $WORK/openclaw_latest

export NODE_VERSION=v22.16.0
case "$(uname -m)" in
  x86_64) NODE_ARCH=x64 ;;
  aarch64|arm64) NODE_ARCH=arm64 ;;
  *) echo "Unsupported arch"; exit 2 ;;
esac

NODE_TAR="$WORK/openclaw_latest/cache/node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz"
if [ ! -s "$NODE_TAR" ]; then
  curl -fL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" -o "$NODE_TAR"
fi

rm -rf "$WORK/openclaw_latest/.local/node"
mkdir -p "$WORK/openclaw_latest/.local/node"
tar -xJf "$NODE_TAR" -C "$WORK/openclaw_latest/.local/node" --strip-components=1

export PATH="$WORK/openclaw_latest/.local/node/bin:$PATH"
node -v
npm -v
```

---

## 5. Install OpenClaw

```bash
cd $WORK/openclaw_latest
export PATH="$WORK/openclaw_latest/.local/node/bin:$PATH"

npm install -g openclaw@latest --prefix "$WORK/openclaw_latest/.local/npm"
ln -sf "$WORK/openclaw_latest/.local/npm/bin/openclaw" "$WORK/openclaw_latest/bin/openclaw"

$WORK/openclaw_latest/bin/openclaw --version
```

---

## 6. Basic Configuration

```bash
export WORK=/lustre/orion/gen150/scratch/zye25/Agentic
cd $WORK/openclaw_latest
export PATH="$WORK/openclaw_latest/.local/node/bin:$PATH"
export OPENCLAW_CONFIG_PATH="$WORK/openclaw_latest/state/openclaw.json"
export VLLM_API_KEY=vllm-local

source "$(ls -1t $WORK/logs/vllm_launcher/*_p8011.endpoints.sh | head -n1)"
export OPENCLAW_BASE_URL="$ENDPOINT_A"
export OPENCLAW_MODEL_ID="meta-llama/Llama-3.1-70B-Instruct"

$WORK/openclaw_latest/bin/openclaw setup --workspace "$WORK/openclaw_latest"

$WORK/openclaw_latest/bin/openclaw config set agents.defaults.workspace "$WORK/openclaw_latest"
$WORK/openclaw_latest/bin/openclaw config set tools.profile full
$WORK/openclaw_latest/bin/openclaw config set tools.exec.security full
$WORK/openclaw_latest/bin/openclaw config set tools.exec.ask off
$WORK/openclaw_latest/bin/openclaw config set tools.web.fetch.enabled true
$WORK/openclaw_latest/bin/openclaw config set tools.web.search.enabled true

$WORK/openclaw_latest/bin/openclaw config set tools.allow '["read","edit","write","exec","process","sessions_list","sessions_history","sessions_send","sessions_spawn","session_status","memory_search","memory_get","web_fetch","subagents"]' --strict-json
```

---

## 7. Configure vLLM Provider

```bash
cd $WORK/openclaw_latest

PROVIDER_JSON=$(cat <<JSON
{
  "baseUrl": "${OPENCLAW_BASE_URL}",
  "apiKey": "${VLLM_API_KEY}",
  "api": "openai-completions",
  "models": [
    {
      "id": "${OPENCLAW_MODEL_ID}",
      "name": "vLLM ${OPENCLAW_MODEL_ID}",
      "reasoning": false,
      "input": ["text"],
      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
      "contextWindow": 32768,
      "maxTokens": 8192,
      "compat": {
        "supportsStore": false,
        "supportsStrictMode": false,
        "supportsDeveloperRole": false
      }
    }
  ]
}
JSON
)

$WORK/openclaw_latest/bin/openclaw config set models.providers.vllm "$PROVIDER_JSON" --strict-json
$WORK/openclaw_latest/bin/openclaw config set agents.defaults.model.primary "vllm/${OPENCLAW_MODEL_ID}"
$WORK/openclaw_latest/bin/openclaw config validate
```

---

## 8. Minimal Smoke Test

```bash
cd $WORK/openclaw_latest
export PATH="$WORK/openclaw_latest/.local/node/bin:$PATH"
export OPENCLAW_CONFIG_PATH="$WORK/openclaw_latest/state/openclaw.json"

RUN_ID=smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$WORK/openclaw_latest/logs/$RUN_ID"

$WORK/openclaw_latest/bin/openclaw agent \
  --local \
  --session-id "$RUN_ID" \
  --message "Read the first doc under/lustre/orion/gen150/scratch/zye25/Agentic/docs/, then write summary to /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_latest/logs/${RUN_ID}/summary.txt, then finish." \
  --timeout 600 \
  --json \
  1> "$WORK/openclaw_latest/logs/$RUN_ID/agent.json" \
  2> "$WORK/openclaw_latest/logs/$RUN_ID/agent.err"

ls -l "$WORK/openclaw_latest/logs/$RUN_ID"
cat "$WORK/openclaw_latest/logs/$RUN_ID/agent.err"
```
