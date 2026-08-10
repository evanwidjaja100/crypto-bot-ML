#!/usr/bin/env bash
# ./run.sh — start the bot in paper mode (background). Safe to run twice.
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$HOME/.local/lib"

PIDFILE=/tmp/opencode/run_paper.pid
LOG=/tmp/opencode/run_paper.log

# Already running? Tell the user and do nothing (prevents double-start).
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Bot is already running (pid $(cat "$PIDFILE"))."
    echo "Watch:   tail -f $LOG"
    echo "Stop:    kill $(cat "$PIDFILE")"
    exit 0
fi

nohup .venv/bin/python scripts/run_bot.py --warmup-bars 2000 > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 2

if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Bot started (pid $(cat "$PIDFILE")) — paper mode, one decision per 60m candle."
    echo "Watch:   tail -f $LOG"
    echo "Trades:  cat data/runner/journal_\$(date -u +%Y%m%d).jsonl"
    echo "Stop:    kill $(cat "$PIDFILE")"
else
    echo "Bot failed to start. Last log lines:"
    tail -5 "$LOG"
    exit 1
fi
