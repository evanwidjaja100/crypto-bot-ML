# Implementation Plan — Production Readiness Remediation

**Source:** [`production-readiness-review.md`](production-readiness-review.md) (findings F1–F22)
**Repo:** `evanwidjaja100/crypto-bot-ML` @ `da8a835` · **Plan written:** 2026-08-13
**Baseline verified:** 133 tests passing; both flagged warnings reproduce; F1 corruption confirmed on disk.

---

## 0. How to read this plan

Nine phases, ordered so each unblocks the next. Phases 1–4 are small diffs that restore trust in
what the bot is measuring. Phases 5–6 are the real lift before keys touch mainnet. Phases 7–9
harden and document.

Every task carries:

- **Files** — exact paths and line anchors against `da8a835`.
- **Change** — what to write, with code sketches that match the real signatures in the repo.
- **Tests** — the assertions that make the fix permanent.
- **Done when** — the observable acceptance criterion.

Effort estimates assume one developer already familiar with the codebase.

### Verified baseline (run before starting)

```bash
wsl -e bash -lc "cd '/mnt/d/Desktop/Coding/crypto bot + ML' && export LD_LIBRARY_PATH=\$HOME/.local/lib && .venv/bin/python -m pytest -q"
```

- 133 tests pass.
- `data/raw/BTCUSDT_60.parquet`: 19,321 rows, closes `$50.80 → $1,999,999.80`, **2,467 bars >10%/h, 1,077 bars >25%/h**, window 2024-05-27 → 2026-08-10. F1 is confirmed, not theoretical.
- `.env` currently has `BYBIT_TESTNET=true`, `BOT_MODE=paper` — the exact combination that produced the corruption.
- Two warnings reproduce: numpy 2.5 shape-deprecation inside joblib, and `RuntimeWarning: overflow encountered in scalar power` at [`src/models/evaluate.py:93`](src/models/evaluate.py#L93).

### Environment note

The venv is a **WSL** venv (`.venv/bin/python`, Python 3.14.4) on a repo that lives on a Windows
drive. Every command in this plan runs inside WSL from `/mnt/d/Desktop/Coding/crypto bot + ML`.
Native Windows Python (3.12.10) has no dependencies installed and the scripts' `sys.path` hack
(F17) breaks there anyway. Do not mix the two.

### Working rules

1. **One phase per branch, one task per commit.** Tag `pre-remediation` at `da8a835` first.
2. **Test-first for every behavioral fix.** Write the failing test, then the fix. The suite is the
   project's best asset; the remediation should end with meaningfully more of it, not the same 133.
3. **Never delete data.** Quarantine, don't `rm`. The corrupted parquet is evidence.
4. **The bot stays stopped** from the start of Phase 1 until Phase 4 lands. Its current state
   (`data/runner/state.json`, journals) is derived from fictional prices and must not be resumed.

---

## Deviation from the review's suggested order (one change, deliberate)

The review puts the dependency lock (F13) at step 7. **Pull it forward to Phase 0.**

Reason: Phase 1 ends with *retrain and re-gate*, producing the model artifact that everything
afterwards is judged against. Producing that artifact in an unpinned, bleeding-edge environment
(pandas 3.0.5, numpy 2.5.1, sklearn 1.9.0, Python 3.14.4) means it may not unpickle after the next
reinstall — and you'd have to retrain again after locking anyway. Locking first costs half a day
and makes the Phase 1 retrain the *last* retrain you need for this cycle.

Everything else follows the review's ordering.

---

## Phase 0 — Preflight and reproducible environment

**Blocks:** everything. **Effort:** ~0.5 day.

### 0.1 · Snapshot the current state

```bash
git tag pre-remediation da8a835
git checkout -b remediation/phase-0-preflight
mkdir -p data/quarantine/2026-08-13
cp data/raw/*.parquet data/quarantine/2026-08-13/
cp -r data/runner data/quarantine/2026-08-13/runner
```

Write `data/quarantine/2026-08-13/README.md` recording *why* these files are quarantined (mixed
testnet/mainnet closes, 1,077 bars >25%/h) so they aren't mistaken for a usable backup later.

The quarantine directory is under `data/`, which `.gitignore` excludes — that's fine, it is local
evidence, not repo content.

### 0.2 · Stop the bot and neutralize the stale session

- Kill the running paper process (`kill $(cat /tmp/opencode/run_paper.pid)`).
- Move `data/runner/state.json` and `data/runner/journal_*.jsonl` into quarantine. They encode a
  ledger built on fictional prices; restarting against them re-imports the corruption into the new
  run.

### 0.3 · Adopt `uv` and commit a lock file *(F13)*

**Files:** `pyproject.toml`, new `uv.lock`, new `.python-version`

- Pin `requires-python = ">=3.11,<3.13"`. Python 3.14 + numpy 2.5 + pandas 3.0 is the combination
  producing today's warnings; 3.12 is the boring choice for a system that must run unattended.
- Tighten the open ranges to compatible-release pins on the majors that break pickles:

```toml
dependencies = [
    "pybit>=5.8,<6",
    "pandas>=2.2,<2.4",
    "numpy>=1.26,<2.2",
    "pyarrow>=14,<19",
    "scikit-learn>=1.4,<1.6",
    "lightgbm>=4.7,<5",
    "pydantic>=2.5,<3",
    "pydantic-settings>=2.1,<3",
    "PyYAML>=6.0,<7",
    "python-dotenv>=1.0,<2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6", "mypy>=1.11"]
```

- `uv lock && uv sync --extra dev`, recreate `.venv` from the lock, commit `uv.lock`.
- Re-run the suite. **Expect drift**: pandas 3 → 2.x may surface behavior differences in
  `pct_change`, `value_counts`, and parquet dtypes. Fix what breaks here, in isolation, before any
  logic changes land — otherwise Phase 1 failures will be indistinguishable from environment
  failures.

### 0.4 · Fix the two known numeric hazards *(F13)*

**File:** [`src/models/evaluate.py:93`](src/models/evaluate.py#L93)

`equity[-1] ** (candles_per_year / len(equity))` overflows on short or degenerate equity curves
(reproduced by `test_proxy_zero_losses_profit_factor_inf`). Guard the exponent:

```python
exponent = candles_per_year / len(equity)
final = float(equity[-1])
if final <= 0.0:
    annualized = -1.0
elif exponent * np.log(final) > 700.0:   # float64 exp ceiling
    annualized = float("nan")            # curve too short to annualize honestly
else:
    annualized = float(final ** exponent - 1.0)
```

Add a test asserting no `RuntimeWarning` is raised for a 5-bar all-win curve, and that the result is
`nan` rather than `inf`.

**Done when:** `uv.lock` is committed, `.venv` is rebuilt from it, `pytest -W error::RuntimeWarning`
passes, and `pip freeze` matches the lock exactly.

---

## Phase 1 — Restore data integrity *(F1)*

**Blocks:** every downstream result. **Effort:** ~1.5 days (plus redownload/retrain wall time).

This is the disqualifying finding. Four code changes, then a full rebuild.

### 1.1 · Make the candle cache network-aware

**File:** [`src/data_ingestion/candle_downloader.py:18-47`](src/data_ingestion/candle_downloader.py#L18-L47)

`CandleStore` is the choke point every writer passes through — `download_data.py`,
`incremental_update`, and `BotRunner._persist_bars`. Enforce provenance there and all three are
covered at once.

```python
CANDLE_COLUMNS = ["ts_ms", "open", "high", "low", "close", "volume", "turnover"]
NETWORKS = ("mainnet", "testnet")

class CandleStore:
    def __init__(self, data_dir: str | Path, *, network: str = "mainnet") -> None:
        if network not in NETWORKS:
            raise ValueError(f"network={network!r} not in {NETWORKS}")
        self.network = network
        self.data_dir = Path(data_dir)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)

    def raw_path(self, symbol: str, interval: str) -> Path:
        # network in the filename: two universes can never share a file
        return self.data_dir / "raw" / f"{symbol}_{interval}_{self.network}.parquet"
```

`write()` gains provenance stamping, a mixed-write refusal, and mandatory validation:

```python
    def write(self, df, symbol, interval, *, validate: bool = True) -> Path:
        df = df.copy()
        if "network" in df.columns:
            foreign = sorted(set(df["network"].dropna().unique()) - {self.network})
            if foreign:
                raise ValueError(
                    f"refusing to write {foreign} rows into the {self.network} store"
                )
        df["network"] = self.network
        df = df.sort_values("ts_ms").drop_duplicates(subset="ts_ms", keep="last").reset_index(drop=True)
        if validate:
            report = validate_candles(df, INTERVAL_MS[interval])
            if not report.ok:
                raise ValueError(f"refusing corrupt write: {report.summary()} -> {report.errors}")
        df[CANDLE_COLUMNS + ["network"]].to_parquet(self.raw_path(symbol, interval), index=False)
        return self.raw_path(symbol, interval)
```

`load()` asserts the loaded rows carry the expected network and raises on mismatch (catches a file
hand-copied into the wrong slot).

**Why filename *and* column:** the filename makes accidental interleaving structurally impossible;
the column makes a hand-moved file detectable. The review offered these as alternatives — do both,
they cost nothing together.

### 1.2 · Market data is mainnet, always

**Files:** [`scripts/run_bot.py:53`](scripts/run_bot.py#L53), [`scripts/download_data.py:39-44`](scripts/download_data.py#L39-L44), [`src/data_ingestion/bybit_client.py:34-43`](src/data_ingestion/bybit_client.py#L34-L43)

Klines are public and need no keys. There is no reason for market data to ever come from testnet.

- `run_bot.py:53` → `BybitClient(testnet=False)`, and `CandleStore(settings.data.data_dir, network="mainnet")`.
- Flip `BybitClient.__init__`'s default to `testnet: bool = False` and document in the docstring
  that testnet klines are for connectivity smoke-tests only and must never reach the mainnet store.
- `download_data.py`: keep `--testnet` but route it to `CandleStore(..., network="testnet")` so it
  writes a physically separate file, and print a loud banner when used.

### 1.3 · Teach the validator to reject impossible price moves

**File:** [`src/data_ingestion/validation.py`](src/data_ingestion/validation.py)

Add two report fields and one check:

```python
@dataclass
class ValidationReport:
    ...
    n_price_jumps: int = 0
    first_jump_ms: int | None = None
    max_bar_move_pct: float = 0.0
```

```python
def validate_candles(df, interval_ms, *, allow_gaps=True, max_bar_move_pct: float | None = 25.0):
    ...
    if max_bar_move_pct is not None and interval_ms <= SPACING_CHECK_MAX_MS and len(d) > 1:
        move = d["close"].pct_change().abs() * 100.0
        report.max_bar_move_pct = float(move.max(skipna=True) or 0.0)
        bad = move[move > max_bar_move_pct]
        if len(bad):
            report.n_price_jumps = len(bad)
            report.first_jump_ms = int(ts.loc[bad.index[0]])
            report.errors.append(
                f"{len(bad)} bar-to-bar close moves > {max_bar_move_pct}% "
                f"(first at ts={report.first_jump_ms}, max {report.max_bar_move_pct:.1f}%)"
            )
```

Include `jumps=` and `max_move=` in `summary()`. Make the threshold configurable as
`data.max_bar_move_pct` in `settings.yaml` (default `25.0`), and add `--allow-jumps` to
`download_data.py` as an explicit, documented escape hatch for a genuine flash-crash bar — the
whole point is that bypassing this is a deliberate operator act, never a silent default.

Blocking-by-default is correct here: on clean mainnet 60m BTC the current data would produce zero
violations, so any hit means something is wrong with the feed.

**Tests:** a frame with a single 300% bar fails validation; the same frame passes with
`max_bar_move_pct=None`; a frame with a legitimate 8% bar passes; D/W/M intervals skip the check.

### 1.4 · Close the two validation bypasses the review didn't name

Two writers reach the parquet **without ever calling `validate_candles`**:

1. [`scripts/download_data.py:73`](scripts/download_data.py#L73) — the non-`--update` path calls
   `store.write(df, ...)` directly. Only `incremental_update` validates.
2. [`src/runner/runner.py:210-219`](src/runner/runner.py#L210-L219) — `_persist_bars` merges freshly
   fetched bars into the cache with no validation at all. **This is the live corruption entry
   point**: the running bot appends whatever the API hands it, forever.

Moving validation into `CandleStore.write` (1.1) fixes both. But the runner needs a *policy* for
what happens when validation fails at runtime, and "raise inside `_persist_bars`" is the wrong
place — the bar has already been traded on by then, since `_process_bar` runs first.

**Validate on ingest, not on persist.** In [`runner.tick()`](src/runner/runner.py#L165-L208), after
the fetch and gap-backfill merge and *before* the `_process_bar` loop:

```python
        new_bars = closed[closed["ts_ms"] > self.last_ts]
        if not new_bars.empty:
            recent = pd.concat([self.ctx.tail(2), new_bars], ignore_index=True)
            report = validate_candles(recent, self.interval_ms,
                                      max_bar_move_pct=self.settings.data.max_bar_move_pct)
            if not report.ok:
                self.gate.kill_switch.trip(f"candle validation failed: {report.errors}")
                raise KillSwitchTripped(...)   # Phase 4 type; RuntimeError until then
```

Splicing the last two context bars in makes the check catch a jump *between* the cached tail and
the first new bar, not just within the new batch. A bad feed halts the bot rather than trading on
it and writing it to disk.

**Tests:** a `FakeClient` returning a bar 300% above the previous one trips the kill switch, writes
nothing to the store, and processes zero bars.

### 1.5 · Wipe, redownload, rebuild, retrain, re-gate

Sequential, each step gated on the previous:

```bash
# 1. old cache is already quarantined (0.1); remove the unkeyed originals
rm data/raw/BTCUSDT_5.parquet data/raw/BTCUSDT_60.parquet data/raw/BTCUSDT_240.parquet

# 2. mainnet redownload (writes BTCUSDT_60_mainnet.parquet)
.venv/bin/python scripts/download_data.py --interval 60 --days 900

# 3. verify integrity before trusting anything downstream
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/raw/BTCUSDT_60_mainnet.parquet")
mv = df["close"].pct_change().abs()
print(f"rows={len(df)} closes={df.close.min():.2f}..{df.close.max():.2f} "
      f"max_move={mv.max()*100:.2f}% jumps>10%={int((mv>0.10).sum())}")
assert (mv > 0.25).sum() == 0, "corruption still present"
assert df["network"].eq("mainnet").all()
PY

# 4. rebuild + retrain + re-gate
.venv/bin/python scripts/build_features.py
.venv/bin/python scripts/train_model.py
.venv/bin/python scripts/backtest.py
```

**Expect the gate to FAIL.** The promoted `BTCUSDT_60_37ec52b2` model's PF 1.06 came from prices
that included 1,077 impossible bars — some of that "edge" was almost certainly the model learning
testnet's thin-book spikes. A FAIL here is the plan working correctly, not a setback. If it fails:
do **not** loosen `min_promote`. Record the clean-data metrics as the new honest baseline and let
Phase 8 (holdout, purge, relabel) do the science before any config is re-tuned.

Also: the old `BTCUSDT_5*` and `BTCUSDT_240*` artifacts are from the abandoned 5m era — quarantine,
don't rebuild.

**Done when:** the mainnet cache has zero >25% bars, the `network` column is uniform, features and
dataset are rebuilt from it, and the retrain verdict (PASS or FAIL) is recorded in the plan's
changelog with its metrics.

---

## Phase 2 — Couple the trading network to the mode *(F2)*

**Effort:** ~0.5 day.

Today `mode` and `BYBIT_TESTNET` are independent. `mode=testnet` + `BYBIT_TESTNET=false` signs
**real mainnet orders** with every log line claiming testnet, and never asks for `ENABLE-LIVE`.

### 2.1 · Derive the network from the mode

**File:** [`src/config.py`](src/config.py)

```python
# Market data is public and always mainnet. Only ORDER endpoints follow the mode.
TRADING_NETWORK_BY_MODE = {
    "backtest": None,      # no orders
    "paper":    None,      # no orders
    "testnet":  "testnet",
    "live":     "mainnet",
}

class Settings(BaseModel):
    ...
    @property
    def market_data_network(self) -> str:
        return "mainnet"

    @property
    def trading_network(self) -> str | None:
        return TRADING_NETWORK_BY_MODE[self.mode]

    @property
    def order_endpoints_testnet(self) -> bool:
        """Value for pybit HTTP(testnet=...). Raises in modes that place no orders."""
        net = self.trading_network
        if net is None:
            raise RuntimeError(f"mode={self.mode} places no orders")
        return net == "testnet"
```

### 2.2 · Delete the env flag as a degree of freedom

Remove `bybit_testnet` from `EnvSettings` ([`src/config.py:107`](src/config.py#L107)) and add a
validator that **fails loudly** if the stale variable is still present, rather than ignoring it:

```python
    @model_validator(mode="after")
    def _reject_legacy_network_flag(self) -> "Settings":
        if os.getenv("BYBIT_TESTNET") is not None:
            raise ValueError(
                "BYBIT_TESTNET is no longer honored — the trading network is derived from mode "
                "(testnet -> testnet, live -> mainnet) and market data is always mainnet. "
                "Delete BYBIT_TESTNET from .env."
            )
        return self
```

Silently ignoring it would leave an operator believing a knob still works. Then remove the line from
`.env` and from `.env.example` (create one if absent — it doesn't exist today).

### 2.3 · Use it at the call sites

**File:** [`scripts/run_bot.py:53, 76-88`](scripts/run_bot.py#L53-L88)

```python
    client = BybitClient(testnet=False)                                   # market data: mainnet
    store = CandleStore(settings.data.data_dir, network=settings.market_data_network)
    ...
    if settings.mode in ("testnet", "live"):
        session = HTTP(
            testnet=settings.order_endpoints_testnet,                     # derived, not env
            api_key=settings.env.bybit_api_key,
            api_secret=settings.env.bybit_api_secret,
            timeout=15,
        )
        log.warning("ORDERS -> %s (mode=%s)", settings.trading_network, settings.mode)
```

The `ENABLE-LIVE` phrase check at [`src/config.py:152`](src/config.py#L152) now genuinely guards
mainnet money, because mainnet order endpoints are reachable only through `mode=live`.

**Tests:**
- `Settings(mode="live")` without the confirm phrase raises (existing behavior, keep the test).
- `Settings(mode="testnet").order_endpoints_testnet is True`; `mode="live"` → `False`.
- `mode="paper"` → `trading_network is None` and `order_endpoints_testnet` raises.
- With `BYBIT_TESTNET` in `os.environ`, `load_settings()` raises with a message naming the variable.

**Done when:** `grep -rn "bybit_testnet" src scripts` returns only the rejection validator.

---

## Phase 3 — Risk state survives restarts *(F3, F4)*

**Effort:** ~1 day.

Two state bugs that make the safety rails weaker than they read.

### 3.1 · Persist the full daily-loss state *(F3)*

**File:** [`src/risk/limits.py:45-81`](src/risk/limits.py#L45-L81)

`restore_day_pnl()` restores only `_pnl`. `_day` stays `None`, so
[`allowed()`](src/risk/limits.py#L73-L77) returns `True` unconditionally and the first close resets
`_pnl` to zero. Lose 2%, restart, trade again with a clean slate. Replace with a full round-trip:

```python
    def snapshot(self) -> dict:
        return {"day": self._day, "pnl": self._pnl, "equity_base": self._equity_base}

    def restore(self, snap: dict) -> None:
        self._day = snap.get("day")
        self._pnl = float(snap.get("pnl", 0.0))
        self._equity_base = float(snap.get("equity_base", self._equity_base))
```

Delete `restore_day_pnl` (only caller is [`runner.py:105`](src/runner/runner.py#L105)). Also fix
`reset()`, which currently leaves `_equity_base` stale — reset it to the constructor value.

**File:** [`src/runner/runner.py:87-110`](src/runner/runner.py#L87-L110)

```python
    def _save_snapshot(self) -> None:
        snap = {
            "last_ts": self.last_ts,
            "broker": self.broker.snapshot(),
            "daily_loss": self.gate.daily_loss.snapshot(),
        }
```

On restore, read `daily_loss`; if only the legacy `daily_loss_pnl` key is present, log a warning
and treat the state as untrusted (start flat) rather than half-restoring it.

**Tests** (the review's exact scenario, plus the boundary):
- Trip the limit → `snapshot()` → fresh tracker → `restore()` → `allowed(same_day_ts)` is `False`.
- After restore, `allowed(next_day_ts)` is `True` (a new day must still reset).
- Runner-level: drive a losing close past the limit, `_save_snapshot()`, build a new `BotRunner` on
  the same `state_path`, warm up, and assert the next entry is rejected with a daily-loss reason in
  the journal.

### 3.2 · Kill-switch tombstone *(F4)*

**File:** [`src/risk/limits.py:7-42`](src/risk/limits.py#L7-L42)

A tripped switch lives only in memory. Anything that restarts the process forgets it — including a
**reconciliation mismatch**, which means the exchange and the ledger disagree about real money.

```python
class KillSwitch:
    def __init__(self, max_api_error_streak: int = 5, *,
                 tombstone_path: str | Path | None = None) -> None:
        ...
        self._tombstone = Path(tombstone_path) if tombstone_path else None
        if self._tombstone is not None and self._tombstone.exists():
            data = json.loads(self._tombstone.read_text())
            self._tripped, self._reason = True, data.get("reason", "unknown")
            self._tripped_at = data.get("tripped_at")

    def trip(self, reason: str) -> None:
        if self._tripped:
            return
        self._tripped, self._reason = True, reason
        self._tripped_at = datetime.now(UTC).isoformat()
        self._write_tombstone()          # atomic tmp + rename, same pattern as the state snapshot

    def reset(self) -> None:
        ...
        if self._tombstone is not None:
            self._tombstone.unlink(missing_ok=True)
```

The tombstone lives beside the state snapshot (`data/runner/KILL_SWITCH.json`), **never in `/tmp`** —
that is exactly the mistake F9 flags. `RiskGate.__init__` takes and forwards `tombstone_path`;
`run_bot.py` passes `Path(args.state_path).parent / "KILL_SWITCH.json"`.

### 3.3 · Refuse to start while the tombstone exists

**File:** [`scripts/run_bot.py`](scripts/run_bot.py), before `warmup()`

```python
    if runner.gate.kill_switch.is_tripped():
        log.error("KILL SWITCH ACTIVE: %s (tripped %s). Investigate, then: "
                  "python scripts/reset_kill_switch.py --reason '<what you fixed>'",
                  runner.gate.kill_switch.describe(), runner.gate.kill_switch.tripped_at())
        return 3
```

Exit code 3 stays the kill-switch code. Phase 5's supervisor uses
`RestartPreventExitStatus=3` so auto-restart can never resurrect a tripped bot — this is the seam
where F4 and F9 meet, and getting it wrong turns the tombstone into a restart loop.

### 3.4 · `scripts/reset_kill_switch.py`

Reset must be a deliberate, attributable operator act:

- Print the tombstone (reason + timestamp) and require an explicit `--reason` describing the fix.
- Require `--yes` to actually proceed.
- Append `{tripped_at, trip_reason, reset_at, reset_reason}` to `data/runner/kill_switch_audit.jsonl`
  before unlinking, so the trip history survives the reset.

**Tests:** trip → construct a new `KillSwitch` on the same path → `is_tripped()` is `True` and the
reason survives; `reset()` removes the file; a `KillSwitch` with `tombstone_path=None` behaves
exactly as today (keeps the existing 133 green); the runner's `_exchange_setup_and_reconcile`
mismatch path writes a tombstone.

**Done when:** trip the switch, `kill -9` the process, restart → it refuses to start and names the
original reason.

---

## Phase 4 — Make the error-streak design actually engage *(F5)*

**Effort:** ~0.5 day.

`BybitClient._request` wraps exhausted retries in `RuntimeError`
([`bybit_client.py:83`](src/data_ingestion/bybit_client.py#L83)); `tick()` re-raises it; `run_bot`'s
`except RuntimeError` exits 3 ([`run_bot.py:117-121`](scripts/run_bot.py#L117-L121)). One network
blip outlasting the client's internal retries kills the bot — and `max_api_error_streak=5` can never
accumulate a streak, because the process dies on error #1. The first `tick()` at line 110 isn't
covered by that handler at all and crashes with a raw traceback.

### 4.1 · A dedicated exception type

**File:** new `src/risk/exceptions.py`

```python
class KillSwitchTripped(RuntimeError):
    """Fatal: the bot must stop and stay stopped until an operator resets it."""
```

Subclassing `RuntimeError` keeps existing `except RuntimeError` sites behaving as before, while
letting the loop order its handlers `KillSwitchTripped` → `Exception`. Replace all four raise sites
in [`runner.py:174, 207, 346, 359`](src/runner/runner.py#L174).

### 4.2 · Restructure the run loop

**File:** [`scripts/run_bot.py:109-124`](scripts/run_bot.py#L109-L124)

```python
    consecutive_failures = 0
    try:
        while True:
            try:
                runner.tick()
                consecutive_failures = 0
            except KillSwitchTripped as exc:
                log.error("HALT — kill switch: %s", exc)
                notifier.alert("kill switch tripped", str(exc))     # Phase 5
                return 3
            except Exception as exc:                                # noqa: BLE001 — transient
                consecutive_failures += 1
                log.warning("tick failed (%d consecutive): %s", consecutive_failures, exc,
                            exc_info=True)
                if consecutive_failures == 3:
                    notifier.alert("repeated tick failures", str(exc))
            if args.once:
                return 0
            time.sleep(args.sleep_secs or (runner.interval_ms / 1000.0))
    except KeyboardInterrupt:
        log.info("interrupted; state snapshot already saved per bar")
        return 0
```

The first tick now runs inside the same handler as every other tick — no more uncovered raw
traceback. `--once` still exits after one tick, but now reports failures instead of exploding.

`gate.on_api_error()` already fires inside `tick()`'s fetch handler
([`runner.py:170-175`](src/runner/runner.py#L170-L175)), so with the loop continuing, five
consecutive failures now trip the switch and *that* exits 3 — the design working as documented.

Wrap `runner.warmup()` in its own try: a warmup failure is a startup failure (exit 1), not a
kill-switch trip.

### 4.3 · Wire the executor's kill-switch callbacks *(finding beyond the review)*

`BybitExecutor` accepts `on_api_error` / `on_api_success` and its docstring says *"The runner wires
these to the RiskGate"* ([`bybit_executor.py:10-12`](src/execution/bybit_executor.py#L10-L12)). It
doesn't. [`run_bot.py:87`](scripts/run_bot.py#L87) constructs `BybitExecutor(session, settings.symbol)`
with no callbacks, so **order-path API failures never reach the streak counter at all**. Fixing F5
without this leaves half the error surface unmonitored.

Add a public binder rather than reaching into privates:

```python
# src/execution/bybit_executor.py
    def bind_callbacks(self, on_api_error, on_api_success) -> None:
        self._on_api_error, self._on_api_success = on_api_error, on_api_success
```

```python
# src/runner/runner.py — BotRunner.__init__, after self.gate is built
        if executor is not None and hasattr(executor, "bind_callbacks"):
            executor.bind_callbacks(self.gate.on_api_error, self.gate.on_api_success)
```

Also pass `max_retries=settings.execution.max_order_retries` at construction — the configured value
is currently ignored in favor of the default 3.

**Tests:**
- A client whose `fetch_candles` raises: `main(["--once"])` returns 0 and logs a warning (not 3).
- Five consecutive failing ticks trip the switch and the sixth returns 3.
- A failing executor request increments the gate's streak (asserts the binding).

**Done when:** a simulated 20-minute API outage leaves the bot alive and trading on recovery, while
five *consecutive* failures halt it.

---

### 🚦 Gate: paper trading is meaningful again

Phases 1–4 complete. Restart paper on clean mainnet data. Results from here measure something real.

---

## Phase 5 — Stand up operations *(F8, F9)*

**Effort:** ~1.5 days.

Every failure mode currently ends as a log line in `/tmp` and a dead process.

### 5.1 · Notifier

**File:** new `src/monitoring/notify.py`

Small interface, one implementation, a no-op default so paper/tests need no configuration:

```python
class Notifier(Protocol):
    def alert(self, title: str, body: str, *, severity: str = "error") -> None: ...

class NullNotifier: ...                        # default; logs only
class TelegramNotifier:                        # bot token + chat id from .env
    # requests.post with a 5s timeout, swallow-and-log on failure —
    # a broken notifier must never take down the bot
```

Config block in `settings.yaml`:

```yaml
monitoring:
  notifier: none          # none | telegram | discord
  heartbeat_url: ""       # healthchecks.io-style ping URL
  heartbeat_every_ticks: 1
```

Credentials (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) go in `.env` via `EnvSettings`, never in
`settings.yaml`.

**Alert on:** kill-switch trip (with reason), reconciliation mismatch, order failure, daily-loss
halt, candle-validation rejection, start, and clean stop. Include mode, symbol, model_id, and
current equity in the body — an alert you can't act on without SSHing in is half an alert.

### 5.2 · Dead-man heartbeat

Ping `heartbeat_url` at the end of each successful `tick()`. Silence then alerts *for* you — the
one failure mode a push notifier can never cover is the bot being dead. Failures to ping are logged
at debug and never propagate.

### 5.3 · Rotating file logs

**File:** [`src/monitoring/logging_setup.py`](src/monitoring/logging_setup.py)

`logging.log_dir` exists in [`config/settings.yaml:91`](config/settings.yaml#L91) and is **never
read**. Logs go to stdout only, unrotated, redirected to `/tmp`, which vanishes on reboot.

```python
def setup_logging(level="INFO", stream=sys.stdout, log_dir: str | Path | None = None) -> None:
    ...
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(Path(log_dir) / "bot.jsonl", when="midnight",
                                      backupCount=30, utc=True)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
```

Pass `settings.logging.log_dir` from all six scripts. Keep the JSON formatter — the journal and logs
stay machine-readable together.

### 5.4 · Supervised deployment

**Recommendation: systemd inside WSL.** Docker adds an image build to every retrain cycle for a
single-process Python bot, and the repo's tooling (`run.sh`, `watch.sh`, the WSL venv) is already
WSL-shaped. Revisit Docker only if the bot moves off this machine.

**File:** new `deploy/crypto-bot.service`

```ini
[Unit]
Description=Crypto bot (paper/testnet)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/mnt/d/Desktop/Coding/crypto bot + ML
Environment=LD_LIBRARY_PATH=%h/.local/lib
ExecStart=/mnt/d/.../.venv/bin/python scripts/run_bot.py --warmup-bars 2000
Restart=on-failure
RestartSec=30
RestartPreventExitStatus=3        # 3 = kill switch. NEVER auto-restart a tripped bot.
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

`RestartPreventExitStatus=3` is the line that makes F4's tombstone meaningful under supervision.
Verify it explicitly: trip the switch, confirm systemd leaves the unit failed rather than looping.

Requires `systemd=true` in `/etc/wsl.conf` (WSL 2, recent builds) plus Windows "WSL on boot" if
start-on-boot is wanted — document both in the README, including the honest caveat that a Windows
reboot needs the WSL instance started before the unit runs.

Retire `run.sh`/`watch.sh` to `scripts/dev/` and document them as WSL-only dev tools (F17).

**Done when:** `systemctl --user restart crypto-bot` works; killing the process auto-restarts it
within 30s; a tripped kill switch leaves it stopped; a Telegram alert arrives for each of the five
trigger conditions; logs land in `./logs/` and rotate.

---

## Phase 6 — Close the live-execution gap *(F6, F7, F20)*

**Effort:** ~3 days plus a 2-week testnet soak. The largest phase; do not compress it.

Everything here is *only* exercised in testnet/live. Paper and backtest semantics must be
byte-identical afterwards — [`tests/test_engine_paper_equivalence.py`](tests/test_engine_paper_equivalence.py)
is the guard, and it must stay green throughout.

### 6.1 · Exchange-native stops and targets *(F6)*

If the process dies holding a position, nothing on the exchange protects it. And even alive, the bot
reacts only at bar close: paper books a stop fill *at the stop price mid-bar*
([`paper_broker.py:204-229`](src/execution/paper_broker.py#L204-L229)) while live sends a market
order after the bar closes at whatever price then prevails. Live diverges from the paper ledger on
every stopped trade, always unfavorably.

**Prerequisite — price sanitization.** [`get_instruments_info`](src/execution/bybit_executor.py#L151-L162)
reads only `lotSizeFilter`. TP/SL prices must be rounded to `priceFilter.tickSize` or Bybit rejects
them. Extend the cached dict with `tick_size` and add `sanitize_price(px, side, kind)` rounding
*conservatively* (stops away from entry, targets toward it).

```python
    def set_trading_stop(self, *, stop_loss: float | None, take_profit: float | None) -> dict:
        kwargs = {"category": "linear", "symbol": self.symbol,
                  "tpslMode": "Full", "positionIdx": 0}
        if stop_loss is not None:
            kwargs["stopLoss"] = str(self.sanitize_price(stop_loss, kind="stop"))
        if take_profit is not None:
            kwargs["takeProfit"] = str(self.sanitize_price(take_profit, kind="target"))
        return self._request("set_trading_stop", **kwargs)
```

**Failure policy — decide now, not during an incident.** If the entry order fills but
`set_trading_stop` fails after retries, the position is naked:

1. Retry up to `max_order_retries`.
2. Still failing → immediately flatten with a `reduce_only` market order.
3. Alert, and trip the kill switch.

An unprotected position is strictly worse than no position. Encode this as a test with a fake
session whose `set_trading_stop` always fails, asserting a reduce-only order is sent and the switch
trips.

**Then demote the client-side check to reconciliation.** In testnet/live, when
`PaperBroker._check_exits` says "stopped out", do not send an exit order — query the exchange
position. If it is flat, the exchange already stopped us; adopt the exchange's actual exit price
into the ledger (6.2). If it still holds a position, that is a genuine mismatch → alert and trip.
The exchange becomes the source of truth for exits in live mode; paper and backtest keep today's
simulated rules unchanged.

Call `cancel_all()` after any close so no stale conditional order survives.

### 6.2 · Adopt real fills, fees, and funding *(F7, F20)*

In testnet/live the broker books fills at *simulated* prices (bar open ± assumed slippage) and sizes
off *simulated* equity. `get_equity()` exists at
[`bybit_executor.py:141`](src/execution/bybit_executor.py#L141) and **is never called**. Every real
deviation — partial fill, fee tier, actual funding — accumulates silently into sizing and daily-loss
math.

```python
    def get_execution(self, order_link_id: str, *, timeout_s: float = 5.0) -> dict | None:
        """Actual execution for an order: avg price, filled qty, cumulative fee.
        Polls briefly — a market IOC fills fast but the execution record lags the ack."""
        # get_executions(category="linear", symbol=..., orderLinkId=...)
        # -> {"avg_price": float, "qty": float, "fee": float, "exec_time_ms": int}
```

Then correct the ledger. Add to `PaperBroker` a live-only amendment that reverses the simulated cash
effect and applies the real one:

```python
    def adopt_actual_fill(self, fill: PaperFill, *, price: float, qty: float, fee: float) -> PaperFill:
        """Live-only: replace a simulated fill's economics with the exchange's.
        Paper and backtest never call this — equivalence tests stay valid."""
```

Invariant to assert in tests: after adopting a fill, `cash + unrealized` equals what it would have
been had the real price/fee been used from the start. Cover open, close, and reverse legs.

**Funding (F20):** replace the constant ±0.01%/8h with the exchange's realized funding in live mode
(available in the execution/transaction log). Keep the constant for backtests and state the
assumption in the README. Book actual funding into `_funding_on_open` so realized P&L — and
therefore the daily-loss limit — reflect real money.

### 6.3 · Continuous reconciliation and exchange equity

Reconciliation runs **once at startup** and compares side/qty only
([`runner.py:361`](src/runner/runner.py#L361)).

- Call `_reconcile_position()` at the end of every `_process_bar` in testnet/live (each bar; the
  extra call is negligible against a 60m interval).
- Add equity reconciliation: compare `executor.get_equity()` against `broker.equity(close)`.
  Config `execution.equity_drift_warn_pct: 1.0` → alert; `equity_drift_trip_pct: 5.0` → trip.
- **In live mode, size off exchange equity**, not simulated equity. Sync the broker's cash to the
  exchange figure at each bar mark before the next decision is made.

### 6.4 · Fix the one-sided position comparison *(finding beyond the review)*

[`runner.py:385`](src/runner/runner.py#L385):

```python
if want is None or side != want[0] or (qty - want[1]) > max(1e-6, want[1] * 0.01):
```

`(qty - want[1])` is **signed**. An exchange position *smaller* than the ledger — the partial-fill
and partial-liquidation case, precisely what F7 warns accumulates — never trips. Must be
`abs(qty - want[1])`. Add a regression test with the exchange holding half the ledger's qty.

### 6.5 · Prove it on testnet

Two weeks minimum, `mode=testnet`, with:
- ≥10 round trips including at least one stop-out and one take-profit.
- A deliberate mid-position `kill -9` → restart → confirm the exchange stop still protected the
  position and reconciliation adopts the true state.
- A deliberate reconciliation mismatch (manually close the position on the exchange) → confirm trip
  + alert + tombstone + no auto-restart.
- A realized-vs-expected report: exchange fills vs. what the paper ledger would have booked.
  Systematic divergence beyond the modeled slippage means 6.1/6.2 aren't done.

---

### 🚦 Gate: testnet with keys

Phases 5–6 proven end-to-end: alerts fire, restarts are safe, exchange-side stops exist.

---

## Phase 7 — CI *(F14)*

**Effort:** ~0.5 day. (F13 landed in Phase 0.)

133 tests — leakage probes, engine/paper equivalence, executor idempotency — that only run when
someone remembers.

**File:** new `.github/workflows/ci.yml`

- Triggers: `push`, `pull_request`.
- `astral-sh/setup-uv`, `uv sync --frozen --extra dev` (fails if `uv.lock` drifts from `pyproject.toml`).
- Steps: `ruff check` → `ruff format --check` → `mypy src scripts` → `pytest -q`.
- Add `[tool.ruff]` (line-length 110 to match the existing style) and `[tool.mypy]` to
  `pyproject.toml`. The code is fully type-annotated, so mypy should be close to clean; start with
  `ignore_missing_imports = true` for `pybit`/`lightgbm` and tighten later.

Land ruff/mypy config and fix their findings in a **separate commit** from the workflow, so the
formatting churn doesn't hide the CI diff.

**Also add:** a `pytest -W error::RuntimeWarning` job so numeric regressions like the
`annualized_return` overflow fail loudly rather than printing.

**Done when:** a red build blocks a PR, and the badge is in the README.

---

## Phase 8 — Tighten the science *(F10, F11, F12)*

**Effort:** ~2 days plus the burn-in window.

Do this **after** Phase 1's clean-data retrain — the metrics need re-establishing anyway.

### 8.1 · An untouched holdout *(F10)*

[`config/settings.yaml:7`](config/settings.yaml#L7) says it plainly: 60m/h=2 is the *"only config
with a real edge"* found by scanning — and the gate validates that survivor on the same test split
the scan consulted. PF 1.06 over 98 trades is inside multiple-comparisons noise.

- Add `--holdout-days N` (default 120) to `build_features.py`. Slice the most recent N days off
  **before** `split_chronological` and save as `holdout.parquet` with a `HOLDOUT_DO_NOT_TOUCH`
  marker file alongside.
- `train_model.py` never loads it — assert this in a test that fails if `holdout` appears anywhere
  in the training path.
- New `scripts/evaluate_holdout.py` is the only reader, and it **appends every use** to
  `artifacts/holdout_uses.jsonl` (timestamp, model_id, git commit, metrics). Peeking becomes
  auditable: three entries for one model means the holdout is spent, and the file says so.
- Roll the holdout forward on each retrain cycle so it stays genuinely unseen.

### 8.2 · Pre-registered paper burn-in

The real promotion gate is out-of-sample time, not another split. Add to `settings.yaml`:

```yaml
promotion:
  burn_in:
    min_weeks: 6
    min_trades: 30
    max_realized_vs_backtest_drift_pct: 30
    min_profit_factor: 1.0
```

Write `scripts/burn_in_report.py` comparing journal-realized P&L against the backtest expectation
over the same window. Criteria are committed **before** the burn-in starts — that is what makes them
pre-registered rather than post-hoc.

### 8.3 · Purge the split boundaries *(F11)*

Labels are computed on the full frame before splitting
([`build_features.py:42-59`](scripts/build_features.py#L42-L59)), so the last `horizon` rows of
train embed prices from val's first bars. At h=2 that's 2 rows per boundary — negligible today,
material if the horizon grows. `walk_forward` already purges correctly
([`walk_forward.py:53`](src/models/walk_forward.py#L53)); the plain split doesn't.

```python
def split_chronological(df, *, train_frac=0.70, val_frac=0.15, purge: int = 0):
    ...
    train = df.iloc[: n_train - purge]              # drop rows whose labels see into val
    val   = df.iloc[n_train : n_train + n_val - purge]
    test  = df.iloc[n_train + n_val :]
```

`build_features.py` passes `purge=settings.labels.horizon`.

**Also fix a latent bug in the same function**: [`dataset.py:48`](src/labels/dataset.py#L48) sets
`"first_ts_ms": {s: int(df["ts_ms"].iloc[0]) for s in SPLITS}` — every split reports the *global*
first timestamp. Dataset metadata is how you audit a split after the fact; it is currently wrong for
val and test.

**Tests:** with `purge=h`, `train.ts_ms.max()` plus `h` intervals is strictly before
`val.ts_ms.min()`; the same for val→test; `purge=0` reproduces today's behavior exactly.

### 8.4 · Past-only label threshold *(F12)*

[`labeler.py:33-37`](src/labels/labeler.py#L33-L37): the threshold is a rolling std of *forward*
returns whose window ends at t, so `fwd_return[t]` participates in setting its own boundary. Not
feature leakage (the threshold isn't a model input), but it makes labels self-referential and
irreproducible at inference.

```python
    fwd_return = close.pct_change(cfg.horizon).shift(-cfg.horizon)
    # threshold from TRAILING realized volatility — fully observable at t
    past_return = close.pct_change(cfg.horizon)
    vol = past_return.rolling(cfg.threshold_window,
                              min_periods=cfg.threshold_window // 2).std()
    thr = (cfg.threshold_sigma * vol).clip(lower=cfg.min_abs_threshold)
```

**Add a leakage probe** to `tests/test_labels.py`, in the style of the existing indicator probes:
mutate `close` at position `t+1..t+h` and assert `label_threshold[t]` does not change. That is the
mechanical statement of "past-only".

### 8.5 · Make label changes invalidate models *(finding beyond the review)*

`feature_set_id` covers features only. Relabeling (8.4) changes the target while leaving `fid`
identical — so [`run_bot.py:66`](scripts/run_bot.py#L66)'s staleness guard, the mechanism that
"refuses to serve a model whose feature_set_id mismatches the live pipeline", **will not notice**.
A model trained on old labels would be served silently against new ones.

Add `label_set_id = sha256(label_params)` computed alongside the feature manifest, store it in model
metadata, and check it in the same guard. Cheap now; a silent, hard-to-diagnose bug later.

---

## Phase 9 — Hygiene *(F15–F22)*

**Effort:** ~1.5 days, parallelizable.

### 9.1 · Model registry with an explicit active pointer *(F15)*

`artifacts/models.json` is 14 legacy strings plus 2 dicts, with duplicates, and "active model" is
whatever was appended last ([`store.py:76-86`](src/models/store.py#L76-L86)) — deployment by side
effect.

```json
{
  "schema": 2,
  "active": {"model_id": "...", "model_type": "lgbm", "promoted_at": "...", "promoted_by": "..."},
  "models": [{"model_id": "...", "model_type": "...", "created_at": "...", "gate_verdict": "PASS"}]
}
```

- `active_model(artifacts_dir)` replaces `latest_model`; `run_bot.py` uses it.
- `promote(model_id, model_type)` is the only writer of `active`; `train_model.py` calls it **only**
  on PASS, preserving the existing "failing models never reach artifacts" property.
- One-shot `scripts/migrate_registry.py` converts the legacy list, dedupes, and points `active` at
  `BTCUSDT_60_37ec52b2` (or whatever Phase 1's retrain produces).
- Record in every model's metadata: git commit (`git rev-parse HEAD`), and versions of python,
  pandas, numpy, scikit-learn, lightgbm, joblib via `importlib.metadata`. Without this, an
  unpickling failure is undiagnosable.
- Prune the superseded 5m-era artifacts to `artifacts/archive/`.

### 9.2 · Batch the cache persistence *(F16)*

`_persist_bars` ([`runner.py:210`](src/runner/runner.py#L210)) loads and rewrites the full year+ of
candles every hour, and Phase 1 adds a full validation pass to each rewrite. Simplest sufficient
fix: buffer new bars in memory and flush every `K` bars (default 24) *and* on clean shutdown; the
per-bar JSONL journal remains the crash-safety net, and warmup's gap backfill already reconstructs
anything unflushed. Partition by month only if the file outgrows that.

### 9.3 · Portability *(F17)*

Replace `sys.path.insert(0, __file__.rsplit("/", 2)[0])` with
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` in **all six** scripts:
`backtest.py`, `benchmark_rulesets.py`, `build_features.py`, `download_data.py`, `run_bot.py`,
`train_model.py`. Move `run.sh`/`watch.sh` to `scripts/dev/` and document the WSL requirement.

### 9.4 · README and runbook *(F18)*

The README is two lines; the design rationale lives in a committed AI-session transcript.

- **README:** architecture sketch, mode semantics (including the new mode→network coupling), safety
  model, setup, the WSL requirement, CI badge, and the stated backtest assumptions (constant
  funding, fill rules).
- **`docs/RUNBOOK.md`:** what to do when the kill switch trips (per reason: API streak,
  reconciliation mismatch, order failure, candle validation); how to reset; how to roll back a
  model via the `active` pointer; how to safely stop and resume; what each alert means and its
  expected response time.
- `session-ses_0250.md` is already deleted in the working tree — commit that, or move it to
  `docs/history/` if the rationale is worth keeping. Don't leave it as an uncommitted deletion.

### 9.5 · Stale docstring *(F19)*

[`paper_broker.py:13-15`](src/execution/paper_broker.py#L13-L15) claims the backtester never arms
cooldown; `src/backtesting/engine.py` arms it on stop-loss exits (confirmed: `if reason in
("stop_loss", "stop_loss_gap"): self._state.cooldown_bars_left = self.risk_cfg.cooldown_bars`). The
code agrees; the comment doesn't. Fix the comment.

### 9.6 · Clock drift *(F21)*

`tick()` classifies bars as closed using `time.time()`
([`runner.py:166`](src/runner/runner.py#L166)); server time is checked only during warmup. Compare
against `client.server_time_ms()` every N ticks (default 12 — hourly bars make this cheap): warn
past 2s, alert past 10s, trip past 60s. A skewed clock silently shifts every decision boundary.

### 9.7 · Secrets hygiene *(F22)*

Nothing is leaked today (verified: `.env` untracked, no key material in history). Keep it mechanical:

- `.pre-commit-config.yaml` with `gitleaks`; add the same as a CI job.
- Commit a `.env.example` with all keys and no values.
- Document in the runbook: when live keys are created — **trade-only permissions, no withdrawal,
  IP-whitelisted**, rotation schedule recorded.

---

### 🚦 Gate: live money

Phase 8's burn-in passed on clean data, with the runbook written and rehearsed.

---

## Findings added beyond the review

Five issues surfaced while grounding this plan against the code. All are folded into the phases
above; listed here so they aren't lost.

| # | Issue | Where | Phase |
| --- | --- | --- | --- |
| A1 | `BybitExecutor`'s kill-switch callbacks are never wired — its docstring claims the runner does it, but [`run_bot.py:87`](scripts/run_bot.py#L87) passes none. Order-path API failures never reach the streak counter. | `run_bot.py`, `bybit_executor.py` | 4.3 |
| A2 | Two cache writers bypass validation entirely: `download_data.py`'s non-`--update` path and `runner._persist_bars`. The latter is the *live* corruption entry point F1 describes. | `download_data.py:73`, `runner.py:210` | 1.4 |
| A3 | Reconciliation's qty check is signed, not absolute: `(qty - want[1]) > tol` never trips when the exchange holds **less** than the ledger — the partial-fill/partial-liquidation case. | `runner.py:385` | 6.4 |
| A4 | `split_chronological` reports the global first timestamp for all three splits, so dataset metadata misstates val and test coverage. | `dataset.py:48` | 8.3 |
| A5 | Relabeling doesn't invalidate models: `feature_set_id` covers features only, so the runner's staleness guard can't detect a label-definition change. Fixing F12 without this is a silent hazard. | `manifest.py`, `run_bot.py:66` | 8.5 |
| A6 | `execution.max_order_retries` is configured but never passed to `BybitExecutor` (default 3 always wins). | `run_bot.py:87` | 4.3 |

---

## Summary

| Phase | Findings | Effort | Blocks |
| --- | --- | --- | --- |
| 0 · Preflight + lock | F13 | 0.5 d | everything |
| 1 · Data integrity | F1, A2 | 1.5 d + rebuild | every result |
| 2 · Network↔mode | F2 | 0.5 d | keys |
| 3 · Risk state persistence | F3, F4 | 1 d | supervision |
| 4 · Error handling | F5, A1, A6 | 0.5 d | unattended run |
| — | **🚦 paper trading is meaningful** | | |
| 5 · Operations | F8, F9 | 1.5 d | testnet keys |
| 6 · Live execution | F6, F7, F20, A3 | 3 d + 2 wk soak | testnet keys |
| — | **🚦 testnet with keys** | | |
| 7 · CI | F14 | 0.5 d | — |
| 8 · Science | F10, F11, F12, A4, A5 | 2 d + burn-in | live money |
| 9 · Hygiene | F15–F22 | 1.5 d | live money |
| — | **🚦 live money** | | |

**Engineering total: ~12.5 days.** Calendar time is dominated by the two waiting periods that can't
be compressed: the 2-week testnet soak (6.5) and the 6-week paper burn-in (8.2). Those two windows —
not the code — are the real distance to live.

### Two things to hold onto

1. **Expect Phase 1's retrain to FAIL the gate.** The current PASS was scored on prices that
   included 1,077 impossible bars. A FAIL is the fix working. Do not respond by loosening
   `min_promote` — respond by letting Phase 8 do the science.

2. **Don't lose the good parts.** Idempotent order placement, backtest/paper fill equivalence, the
   single risk-approval gate, causal feature discipline, promotion gating, atomic snapshots. Every
   phase above must leave those intact —
   [`test_engine_paper_equivalence.py`](tests/test_engine_paper_equivalence.py) and
   [`test_executor.py`](tests/test_executor.py) staying green is the check that it did.

---

## Changelog

### 2026-08-13 — Phase 0 complete (branch `remediation/phase-0-preflight`)

- Tag `pre-remediation` at `da8a835`; quarantine under `data/quarantine/2026-08-13/` (3 corrupted
  parquets + runner state/journals; bot was not running).
- `uv` adopted: `requires-python = ">=3.11,<3.13"`, deps pinned to pickle-stable majors,
  `uv.lock` committed, `.venv` rebuilt on CPython 3.12.13 (pandas 2.3.3, numpy 2.1.3,
  sklearn 1.5.2, lightgbm 4.7.0, pyarrow 18.1.0). No suite drift from the downgrade.
- 0.4: `annualized_return` overflow guarded (nan instead of inf/overflow); regression test added.
- Suite: 133 -> 134 tests, `pytest -W error::RuntimeWarning` green.

### 2026-08-13 — Phase 1 complete (branch `remediation/phase-1-data-integrity`)

- 1.1 network-keyed `CandleStore` (filename + per-row column, mixed-write refusal, mandatory
  validation in `write()`, legacy/unkeyed files rejected on load).
- 1.2 market data always mainnet; `download_data.py --testnet` routes to a separate testnet store
  with a loud banner.
- 1.3 `max_bar_move_pct` (default 25.0) blocking check in `validate_candles`, configurable via
  `data.max_bar_move_pct`, `--allow-jumps` escape hatch.
- 1.4 runner validates on ingest (splicing cached tail) and trips the kill switch on a bad feed
  before trading or persisting.
- 1.5 wipe + mainnet redownload: `BTCUSDT_60_mainnet.parquet` 21,600 rows (2024-02-25 ->
  2026-08-13), closes 49,786..125,981, max bar move 5.09%, zero >10% jumps, network uniform.
- **Retrain verdict: FAIL (as predicted).** Clean-data engine gate: ret 0.0073, pf inf, n_trades 1
  (min 30). The previous promoted model `BTCUSDT_60_37ec52b2` backtested on clean data:
  total_return -11.63%, PF 0.428, 132 trades, 77 stop-loss exits. `min_promote` untouched;
  Phase 8 does the science.
- Suite: 134 -> 145 tests.
