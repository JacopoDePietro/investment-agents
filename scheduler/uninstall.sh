#!/bin/bash
PLIST="$HOME/Library/LaunchAgents/com.investment.agents.plist"
launchctl unload "$PLIST" && rm "$PLIST" && echo "Schedule removed"
