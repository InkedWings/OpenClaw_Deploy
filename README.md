# OpenClaw Deploy on Frontier

This repository is for deploying and running OpenClaw on OLCF Frontier **compute nodes**.

It is **not** designed as a long-running resident service. The typical workflow is:
1. Request a Frontier allocation.
2. Start vLLM/OpenClaw inside that allocation.
3. Run your tasks/tests.
4. Stop processes and release the allocation.

## Repo Layout

- `SELF_DEPLOY_TUTORIAL.md`: main deployment and smoke-test tutorial.
- `script/`: helper scripts (for example, backend start/stop/status).
- `logs/`: runtime logs produced during launches/tests.
- `state/`: local runtime/config state.
- `cache/`: local caches/downloaded artifacts.
- `bin/`: local executable links/wrappers.
- `.local/`: local runtime dependencies (Node/npm, etc.).
- `.openclaw/`: OpenClaw-generated local data.

## Notes

- This repo targets a batch/HPC usage model, not a 24/7 hosted endpoint.
- For complete setup steps, follow `SELF_DEPLOY_TUTORIAL.md`.
