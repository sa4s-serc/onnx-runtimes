#!/usr/bin/env bash
# Auto-resume the measurement campaign after a reboot.
# Installed as an @reboot user cron job. Safe to run anytime:
#   - exits if measure.sh is already running (no double-launch)
#   - exits if all 2160 runs are already recorded
#   - re-applies the sudo-gated CPU prep (reboot resets these)
# measure.sh itself sets governor/turbo/RAPL; only perf_event_paranoid is added here.
set -u

REPO="/home/arihanttr/Desktop/onnx-runtimes"
LOG="$REPO/logs/campaign.log"
TARGET=2160

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.rbenv/bin:$HOME/.rbenv/shims:$PATH"
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"

# already running? do not double-launch (would corrupt measurements.json)
if pgrep -f "bash .*measure\.sh" >/dev/null 2>&1; then
    exit 0
fi

# all done?
DONE=$(jq length "$REPO/data/measurements.json" 2>/dev/null || echo 0)
if [ "$DONE" -ge "$TARGET" ]; then
    exit 0
fi

# perf needs a permissive paranoid level; measure.sh handles RAPL/governor/turbo
sudo -n sysctl -w kernel.perf_event_paranoid=1 >/dev/null 2>&1 || true

echo "===== AUTO-RESUME (boot) $(date '+%F %T') — ${DONE}/${TARGET} done =====" >> "$LOG"
setsid nohup bash "$REPO/measure.sh" >> "$LOG" 2>&1 < /dev/null &
