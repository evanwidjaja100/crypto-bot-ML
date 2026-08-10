#!/usr/bin/env bash
# ./watch.sh — friendly one-screen summary of the paper bot.
# Run once: ./watch.sh    |    Live-update once a minute: watch -n 60 ./watch.sh
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$HOME/.local/lib"
.venv/bin/python - <<'PY'
import json, os, re
from datetime import datetime, timezone
from pathlib import Path

def human(ts):
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def pct(x):
    return f"{x * 100:.1f}%"

def run():
    print("================ PAPER BOT — status ================")

    pidfile = Path("/tmp/opencode/run_paper.pid")
    if pidfile.exists():
        pid = pidfile.read_text().strip()
        alive = os.path.isdir(f"/proc/{pid}")
        if alive:
            print(f"   Bot:      RUNNING (pid {pid})")
        else:
            print(f"   Bot:      NOT RUNNING  (restart with ./run.sh)")
    else:
        print(f"   Bot:      NOT RUNNING  (restart with ./run.sh)")

    log = Path("/tmp/opencode/run_paper.log")
    model_line = ""
    if log.exists():
        m = re.search(r"model: ([^ ]+)", log.read_text())
        if m:
            model_line = m.group(1)
    print(f"   Model:    {model_line or 'none'}")

    cfg = Path("config/settings.yaml")
    mode = "?"
    if cfg.exists():
        mm = re.search(r"^mode:\s*(\w+)", cfg.read_text(), re.M)
        if mm:
            mode = mm.group(1)
    print(f"   Mode:     {mode}" + (" (fake money)" if mode == "paper" else ""))

    # ---- load journal -------------------------------------------------------
    files = sorted(Path("data/runner").glob("journal_*.jsonl"))
    records = []
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    records.sort(key=lambda r: r.get("ts_ms", 0))

    # account value = latest hourly mark, if any
    marks = [r for r in records if r.get("type") == "mark"]
    if marks:
        eq = marks[-1]["equity"]
        print(f"   Account:   ${eq:,.2f}   (last hourly auto-check)")
    else:
        print(f"   Account:   $10,000.00   (initial; no marks yet)")

    # open position, from state.json
    state_file = Path("data/runner/state.json")
    if state_file.exists():
        try:
            st = json.loads(state_file.read_text())["broker"]["state"]
            d = st["direction"]
            if d == 1:
                print(f"   Position:  LONG  size={st['qty']} @ {st['entry_price']:.0f}   stop={st.get('stop_price')} target={st.get('target_price')}")
            elif d == -1:
                print(f"   Position:  SHORT size={st['qty']} @ {st['entry_price']:.0f}   stop={st.get('stop_price')} target={st.get('target_price')}")
            else:
                print(f"   Position:  none")
        except Exception:
            print(f"   Position:  unknown")

    fills = [r for r in records if r.get("type") == "fill"]
    print(f"   Trades:    {len(fills)} total")

    decisions = [r for r in records if r.get("type") == "decision"]
    if decisions:
        d = decisions[-1]
        a = d["action"]
        why = d.get("reasons", "")
        print(f"   Last call: {human(d['ts_ms'])}  {a}   ({why})")

    if fills:
        f = fills[-1]
        pnl = f.get("realized_pnl")
        pnl_s = f"{pnl:+.2f}" if isinstance(pnl, (int, float)) else "?"
        print(f"   Last trade:{human(f['ts_ms'])}  {f['action']} @ {f['price']}  fee ${f.get('fee',0):.2f}  P&L {pnl_s}")
    else:
        print(f"   Last trade: none yet — first trades appear roughly every 3 weeks")

    # outgoing trades summary
    if fills:
        total_pnl = sum(f.get("realized_pnl") or 0.0 for f in fills if isinstance(f.get("realized_pnl"), (int, float)))
        total_fee = sum(f.get("fee") or 0.0 for f in fills)
        print(f"   Lifetime:   realized P&L ${total_pnl:+.2f}  fees ${total_fee:.2f}")
    print("=" * 48)

run()
PY
