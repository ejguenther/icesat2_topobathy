#!/usr/bin/env bash
# Unattended overnight run: ATL08 -> screen -> select -> ATL03 (Harmony) in
# parallel with offset estimation (--watch). Every step is resumable; re-run
# this script after an interruption.
#
# Usage (from projects/geolocation):
#   setsid nohup ./run_poland500.sh local/config_poland500.yaml > /dev/null 2>&1 &
#   uv run python status.py --config local/config_poland500.yaml
set -euo pipefail
CFG=${1:?config path}
cd "$(dirname "$0")"
WORK=$(uv run python -c "from phoreal.config import load_config; print(load_config('$CFG')['work_dir'])")
LOGS="$WORK/logs"; mkdir -p "$LOGS"
log() { echo "[$(date +%T)] $*" | tee -a "$LOGS/driver.log"; }

log "waiting for any running ATL08 download"
while pgrep -f "poland500.py atl08" > /dev/null; do sleep 60; done
log "ATL08 download (resume pass)"
uv run python poland500.py atl08 --config "$CFG" >> "$LOGS/atl08_download.log" 2>&1

if [ ! -f "$WORK/granules/granules.parquet" ]; then
  log "screen"
  uv run python poland500.py screen --config "$CFG" > "$LOGS/screen.log" 2>&1
  log "select"
  uv run python poland500.py select --config "$CFG" > "$LOGS/select.log" 2>&1
fi

rm -f "$LOGS/atl03.done"
log "ATL03 Harmony subsets (background)"
( uv run python poland500.py atl03 --config "$CFG" >> "$LOGS/atl03.log" 2>&1 ; touch "$LOGS/atl03.done" ) &

log "offsets (watch mode)"
uv run python run_offsets.py --config "$CFG" --watch --done-flag "$LOGS/atl03.done" \
  >> "$LOGS/offsets.log" 2>&1
log "retry errors"
uv run python run_offsets.py --config "$CFG" --retry-errors >> "$LOGS/offsets.log" 2>&1
log "review"
uv run python pilot_review.py --config "$CFG" > "$LOGS/review.log" 2>&1 || log "review failed (see review.log)"
log "finished"
touch "$LOGS/run.finished"
