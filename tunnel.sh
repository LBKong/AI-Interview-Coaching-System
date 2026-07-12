#!/usr/bin/env bash
set -euo pipefail
# 需先另开一个终端跑 ./run.sh（本地 8000），再跑本脚本拿公网链接
exec cloudflared tunnel --url http://localhost:8000
