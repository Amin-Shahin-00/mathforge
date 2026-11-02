#!/usr/bin/env bash
# Keep MathForge running around the clock: if the process ever exits
# (crash, out-of-memory, reboot of a dependency), restart it. All progress is
# checkpointed in the run directory, so every restart resumes where it left off.
#
#   bash scripts/run_forever.sh configs/laptop.yaml
#   nohup bash scripts/run_forever.sh configs/gpu.yaml > forever.log 2>&1 &
#
# For a server, the systemd unit in deploy/mathforge.service is more robust.
set -uo pipefail
CONFIG="${1:-configs/laptop.yaml}"
shift || true

while true; do
  echo "$(date '+%F %T')  starting mathforge with $CONFIG"
  mathforge run --config "$CONFIG" "$@"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "$(date '+%F %T')  mathforge finished normally (max_rounds reached)."
    break
  fi
  echo "$(date '+%F %T')  mathforge exited with code $code; restarting in 30 s (Ctrl+C to stop)"
  sleep 30
done
