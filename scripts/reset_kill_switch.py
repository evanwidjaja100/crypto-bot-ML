"""Deliberately, attributably reset a tripped kill switch.

Usage:
    python scripts/reset_kill_switch.py --reason "fixed reconciliation: closed orphan position"

Requires both --reason (what you fixed) and --yes. Appends the trip + reset
pair to the audit log before removing the tombstone, so the trip history
survives the reset.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_TOMBSTONE = Path("data/runner/KILL_SWITCH.json")
DEFAULT_AUDIT = Path("data/runner/kill_switch_audit.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tombstone", default=DEFAULT_TOMBSTONE, type=Path)
    parser.add_argument("--audit", default=DEFAULT_AUDIT, type=Path)
    parser.add_argument("--reason", required=True, help="what you fixed before resetting")
    parser.add_argument("--yes", action="store_true", help="actually proceed")
    args = parser.parse_args(argv)

    if not args.tombstone.exists():
        print(f"no tombstone at {args.tombstone} — the kill switch is not tripped")
        return 0

    data = json.loads(args.tombstone.read_text(encoding="utf-8"))
    print(f"tombstone: reason={data.get('reason')!r} tripped_at={data.get('tripped_at')}")
    if not args.yes:
        print("nothing changed; re-run with --yes to reset")
        return 1

    record = {
        "tripped_at": data.get("tripped_at"),
        "trip_reason": data.get("reason"),
        "reset_at": datetime.now(UTC).isoformat(),
        "reset_reason": args.reason,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with open(args.audit, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    args.tombstone.unlink()
    print(f"kill switch reset; trip recorded in {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
