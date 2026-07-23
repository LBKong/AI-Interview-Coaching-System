#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 离线模式：用本地缓存的 embedding 模型，不去 HuggingFace 做联网检查。
# 大陆网络下 HuggingFace 常连不上，否则首次检索会卡在联网检查上。
# 前提：模型缓存已预热（~/.cache/huggingface/）——离线模式下缓存缺失会直接报错。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
