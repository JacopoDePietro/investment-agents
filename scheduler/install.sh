#!/bin/bash
# macOS only: installs a launchd job that runs the full pipeline Monday-Friday at 22:30 local time.
# Linux: use cron instead, e.g.  30 22 * * 1-5  cd /path/to/repo && .venv/bin/python run.py
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.investment.agents.plist"
DAYS=""
for d in 1 2 3 4 5; do
  DAYS="$DAYS<dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>22</integer><key>Minute</key><integer>30</integer></dict>"
done
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.investment.agents</string>
  <key>ProgramArguments</key><array><string>$P/.venv/bin/python</string><string>$P/run.py</string></array>
  <key>WorkingDirectory</key><string>$P</string>
  <key>StartCalendarInterval</key><array>$DAYS</array>
  <key>StandardOutPath</key><string>$P/logs/launchd.out</string>
  <key>StandardErrorPath</key><string>$P/logs/launchd.err</string>
</dict></plist>
PL
mkdir -p "$P/logs"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST" && echo "Schedule installed: weekdays 22:30"
echo "Tip: to run even when the Mac sleeps:  sudo pmset repeat wakeorpoweron MTWRF 22:25:00"
