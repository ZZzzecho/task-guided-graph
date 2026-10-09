#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
run_log="outputs/patent_longtext_graph_v063.launch.log"
run_pid="outputs/patent_longtext_graph_v063.pid"
if [[ -e "$run_log" || -e "$run_pid" ]]; then
  echo "Launch files already exist. Inspect the existing run before starting another one."
  exit 1
fi
nohup python -u -m scripts.run_patent_longtext_graph "$@" > "$run_log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$run_pid"
echo "Started graph-only pipeline, PID=$task_pid"
echo "Launch log: $run_log"
echo "Stage logs: outputs/patent_longtext_graph_n2048_b4096_k6_v063/{prepare,build,graph}.log"
echo "No GLM/LoRA/GRPO training will be started."
