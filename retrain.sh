#!/usr/bin/env bash
# ./retrain.sh — full retrain: fresh data -> features -> train (gate) -> backtest.
# Ends with a clear PASS or FAIL. If FAIL: the old model stays in place and the
# bot must NOT be (re)started — train_model.py no longer saves failing models.
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$HOME/.local/lib"

echo "== 1/4: updating market data =="
.venv/bin/python scripts/download_data.py --update || { echo "download failed — stopping."; exit 1; }

echo "== 2/4: building features + labels =="
.venv/bin/python scripts/build_features.py || { echo "build failed — stopping."; exit 1; }

echo "== 3/4: training + promotion gate =="
.venv/bin/python scripts/train_model.py
TRAIN_EXIT=$?
if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo
    echo "*******************************************************"
    echo "*  FAIL — the new model did NOT pass the gate.         *"
    echo "*  Do NOT run ./run.sh with this model.                *"
    echo "*  Good news: nothing was saved — the previous good    *"
    echo "*  model is still in place, so a running bot keeps     *"
    echo "*  trading safely. Retry after more data accumulates.  *"
    echo "*******************************************************"
    exit 1
fi

echo "== 4/4: final backtest on the held-out test data =="
.venv/bin/python scripts/backtest.py || { echo "backtest failed — stopping."; exit 1; }

echo
echo "PASS — the new model passed the gate and is saved."
echo "It becomes active the next time the bot starts: ./run.sh"
