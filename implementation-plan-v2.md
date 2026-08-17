# Implementation Plan v2 — Post-Audit Remediation

**Source:** [Production Readiness Audit @ `886e56f`](https://claude.ai/code/artifact/63b372c1-faf8-4c53-b9ab-c73213517014), against [`production-readiness-review.md`](production-readiness-review.md) (F1–F22)
**Repo:** `evanwidjaja100/crypto-bot-ML` @ `886e56f` · **Plan written:** 2026-08-13
**Baseline verified:** 153 pass / 9 fail on Windows (162 total); 5 failures are two real cross-platform bugs, 4 are LightGBM-on-Windows and untriaged.

---

## 0. What this file supersedes

[`implementation-plan.md`](implementation-plan.md) Phases **0–4 are complete** and their changelog is
the record of what shipped. Nothing here re-opens them.

**This file replaces that plan's Phases 5–9.** Same tasks in large part — reordered, plus seven
defects the first plan didn't catch. Read this one for what to do next; read the old one for how
Phases 0–4 were done and why.

Every task carried forward is mapped explicitly below, so nothing from v1 is silently dropped.

| v1 task | Where it went | Why moved |
| --- | --- | --- |
| 5.1–5.4 Operations | **Phase 8** | Don't build monitoring for a strategy that doesn't work yet |
| 6.1–6.3, 6.5 Live execution | **Phase 9** | Same — and gated behind the science |
| 6.4 One-sided position comparison | **5.5** (pulled forward) | One-line safety fix; no reason it waits for Phase 9 |
| 7 CI | **Phase 6** (pulled forward) | Half a day, and it protects every refactor in Phase 7 |
| 8.1–8.2 Holdout, burn-in | **7.4, 7.6** | Now the blocking phase, not the second-to-last |
| 8.3 Boundary purge + `first_ts_ms` | **7.1** | Unchanged |
| 8.4 Past-only threshold | **7.2** | Unchanged |
| 8.5 `label_set_id` | **7.3** | Unchanged |
| 9.1 Registry active pointer | **5.1** (pulled forward) | It is the mechanism keeping a losing model deployed |
| 9.2–9.8 Hygiene | **Phase 10** | Unchanged |

### Why the order changed

v1 sequenced the science last because, at the time, the promoted model looked like it had a small
edge (PF 1.06). Phase 1 removed the corrupted data that produced that number, and the honest
retrain **failed the gate**. The strategy question is no longer a tidy-up at the end — it is the
thing that decides whether Phases 8–10 are worth doing at all.

So: fix what is broken or misleading today (Phase 5), make the suite enforceable (Phase 6), then
answer the edge question (Phase 7) before spending days on operations and live execution.

### Working rules

Unchanged from v1: one phase per branch, one task per commit, test-first for every behavioral fix,
never delete data — quarantine it.

One addition: **the bot stays in paper mode for the whole of this plan.** Phase 9 is the first time
keys are needed, and its gate is explicit.

---

## Phase 5 — Correctness, honesty, and a suite that runs

**Effort:** ~1 day.

Seven small fixes. Three are cross-platform bugs that break the bot outside WSL, one is a safety
control that is half-blind, and one stops a model the project already proved unprofitable from
being the default thing it trades.

### 5.1 · Retire the losing model; make `active` explicit *(F15)*

**Files:** [`src/models/store.py:76-86`](src/models/store.py#L76-L86), `artifacts/models.json`, [`scripts/run_bot.py:47-52`](scripts/run_bot.py#L47-L52)

`latest_model()` defines the deployed model as `entries[-1]`. That resolves to
`BTCUSDT_60_37ec52b2_20260810_074715` — trained on the testnet-contaminated cache, never replaced
because the clean-data retrain correctly refused to promote. Its metadata still reads
`"gate_verdict": "PASS"`; the clean-data backtest saved beside it reads −11.63% and PF 0.428.

The gate can only ever *append*. When nothing passes, the last thing that did — under bad data —
stays wired up. Fix the registry shape so "nothing is deployable" is representable:

```python
# artifacts/models.json
{
    "active": null,  # or a model_id; null means nothing is deployable
    "models": [
        {
            "model_id": "...",
            "model_type": "lgbm",
            "created_at": "...",
            "gate_verdict": "FAIL",
            "git_commit": "...",
            "lib_versions": {...},
        }
    ],
}
```

```python
def active_model(artifacts_dir) -> tuple[object, dict] | None:
    """The explicitly promoted model, or None. Never guesses from ordering."""
```

Migrate the existing file: dedupe the 14 bare-string entries, keep them as history, set
`"active": null`. Replace the `latest_model` call site with `active_model`; `run_bot.py` already
returns exit 2 when no model is found, which is the correct behavior here.

**Tests:**
- A registry whose newest entry has `gate_verdict: FAIL` yields `active_model() is None`.
- A legacy list-shaped registry loads without raising and reports no active model.
- `run_bot.py` exits 2 when `active` is null.

**Done when:** `python scripts/run_bot.py --once` exits 2 with "no promotable model", and no code
path selects a model by list position.

### 5.2 · Atomic snapshots on every platform

**File:** [`src/runner/runner.py:101-103`](src/runner/runner.py#L101-L103)

```python
tmp = self.state_path.with_suffix(".tmp")
tmp.write_text(json.dumps(snap, default=str))
tmp.rename(self.state_path)  # os.rename — fails on Windows if target exists
```

`Path.rename` is an atomic overwrite on POSIX and a hard error on Windows once the destination
exists. The first snapshot succeeds; every subsequent one raises `FileExistsError`, which escapes
`tick()`, is swallowed by the run loop's generic handler, and repeats forever. Four tests fail on
this today.

`KillSwitch._write_tombstone` in [`src/risk/limits.py:70-83`](src/risk/limits.py#L70-L83) already
does it correctly. Match it:

```python
tmp.write_text(json.dumps(snap, default=str), encoding="utf-8")
os.replace(tmp, self.state_path)  # atomic overwrite on POSIX and Windows
```

**Tests:** call `_save_snapshot()` twice in a row and assert the second succeeds and the file
contains the second payload. (This is the assertion the four failing runner tests are missing —
they only fail incidentally.)

**Done when:** `tests/test_runner.py` passes on Windows.

### 5.3 · Explicit UTF-8 on every text read and write

**Files:** [`src/config.py:210`](src/config.py#L210) and 15 further sites — [`src/models/store.py`](src/models/store.py) (47, 50, 52, 61, 72, 82), [`src/features/manifest.py`](src/features/manifest.py) (44, 49), [`src/labels/dataset.py`](src/labels/dataset.py) (71, 80), [`src/risk/limits.py:32`](src/risk/limits.py#L32), [`src/runner/runner.py`](src/runner/runner.py) (92, 102, 109), [`scripts/backtest.py:108`](scripts/backtest.py#L108), [`scripts/reset_kill_switch.py`](scripts/reset_kill_switch.py) (34, 47)

`load_settings` reads the YAML with `path.read_text()` and no encoding, so Python uses the platform
default codec. Under `cp950` the em-dash in `settings.yaml`'s own comment on line 7 makes startup
fail with a decode error before any mode check runs:

```
'cp950' codec can't decode byte 0xe2 in position 225: illegal multibyte sequence
```

The JSON sites mostly survive because `json.dumps` escapes non-ASCII by default — but that is luck,
not design, and the journal is one non-ASCII rejection reason away from the same failure. Pass
`encoding="utf-8"` everywhere, including the two bare `open()` calls.

**Tests:** a config test that round-trips a YAML file containing non-ASCII under a monkeypatched
`locale.getpreferredencoding`, asserting it parses.

**Done when:** `grep -rn "read_text()\|write_text(\|[^.]open(" src scripts | grep -v encoding`
returns nothing.

### 5.4 · `--once` must report failure in its exit code

**File:** [`scripts/run_bot.py:134-139`](scripts/run_bot.py#L134-L139)

```python
            except Exception as exc:
                consecutive_failures += 1
                log.warning("tick failed (%d consecutive): %s", ...)
            if args.once:
                return 0                      # returns 0 even though the tick just failed
```

v1's 4.2 intended `--once` to "report failures instead of exploding" — it reports them to the log,
but the exit code cannot distinguish a clean tick from a crashed one. That matters because the
Phase 4 sign-off rested on `--once` exiting 0, and any CI smoke check (Phase 6) would inherit the
same blindness.

```python
            if args.once:
                return 0 if consecutive_failures == 0 else 1
```

**Tests:** a runner stubbed to raise on `tick()` makes `main(["--once"])` return non-zero; the
happy path still returns 0.

**Done when:** the Phase 4 gate command is re-run and its exit code means something.

### 5.5 · Reconciliation must be symmetric *(v1 6.4)*

**File:** [`src/runner/runner.py:415`](src/runner/runner.py#L415)

```python
        if want is None or side != want[0] or (qty - want[1]) > max(1e-6, want[1] * 0.01):
```

`(qty - want[1])` is signed. An exchange position *larger* than the ledger trips the switch; one
that is *smaller* — partial fill, partial liquidation, a manual close — never does. That is the
direction that costs money: the bot keeps sizing, stopping and computing daily loss against a
position it does not fully hold, and its reduce-only closes silently under-close.

Use `abs(qty - want[1])`. The only existing reconciliation test
([`tests/test_runner.py:389`](tests/test_runner.py#L389)) covers the `want is None` branch, so this
direction has never been exercised.

**Tests:** exchange holds half the ledger's qty → kill switch trips, tombstone written. Exchange
holds 1.005× the ledger's qty (inside tolerance) → no trip.

**Done when:** both directions are covered and the asymmetry cannot return.

### 5.6 · Wire the position-count check to reality

**Files:** [`src/risk/gate.py:49`](src/risk/gate.py#L49), [`src/runner/runner.py:307`](src/runner/runner.py#L307), [`src/runner/runner.py:337`](src/runner/runner.py#L337)

`approve_entry` rejects when `open_positions >= max_open_positions`, but both call sites pass the
literal `0`. With the limit at 1 the comparison is always false — the check reads as an enforced
limit while enforcing nothing.

Harmless today because the paper broker is structurally single-position, which is exactly why it
will be trusted later when that stops being true. Pass
`1 if self.broker.direction != 0 else 0`.

**Tests:** with a position open, a second entry approval is rejected with the max-positions reason.

**Done when:** no call site passes a hardcoded position count.

### 5.7 · Make the suite collect under plain `pytest`

**Files:** [`pyproject.toml`](pyproject.toml), [`tests/conftest.py`](tests/conftest.py)

Nothing puts the repo root on `sys.path`: `conftest.py` does no path setup, and
`[tool.setuptools.packages.find] where = ["src"]` installs the *contents* of `src/` as top-level
packages, not a `src` package. Plain `pytest` fails to collect all 19 modules with
`ModuleNotFoundError: No module named 'src'`; `python -m pytest` works only because it adds the
working directory.

Cheapest correct fix — make the intent explicit rather than relying on invocation:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
pythonpath = ["."]        # requires pytest >= 7
```

**Done when:** bare `pytest` collects 162 tests from a clean shell. Phase 6 depends on this.

---

### 🚦 Gate: the bot no longer trades a known-bad model

`run_bot.py` refuses to start without an explicitly promoted model, the suite is green on both
platforms, and the reconciliation blind spot is closed. Nothing below is safe to evaluate until
this holds.

---

## Phase 6 — CI *(F14)*

**Effort:** ~0.5 day. Pulled forward from v1 Phase 7.

162 tests — including the leakage probes and the engine/paper equivalence invariant — currently run
only when someone remembers, and only when invoked the right way. Phase 7 rewrites the label and
split code those probes exist to protect. CI first.

**File:** new `.github/workflows/ci.yml`

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest -q
```

The Windows leg is not ceremony — it is what catches 5.2 and 5.3 regressing, and the repo lives on
a Windows drive. Add `ruff` and `mypy` configuration to `pyproject.toml`; the code is fully
type-annotated already, so expect a short, real backlog on the first `mypy` run rather than a clean
pass.

Do **not** add a `run_bot.py --once` smoke job until 5.4 lands — before that its exit code proves
nothing.

**Done when:** a red suite blocks a merge on both operating systems, and the badge is in the README.

---

## Phase 7 — Settle the science *(F10, F11, F12)*

**Effort:** ~1 week of work, then 4–8 weeks of burn-in.

This is the phase that decides whether the project continues. Everything below Phase 7 is
infrastructure for a strategy that has not yet been shown to exist.

**The honest starting position.** On clean mainnet data the promoted model produced
`total_gross_pnl = −$928` over 132 trades *before* $250 of fees. The problem is not that costs eat
a thin edge — the directional calls are worse than random, and fees make a losing signal lose
faster. Any framing that treats this as a cost-optimization problem is starting from the wrong
premise.

Tasks 7.1–7.3 remove known defects from the labeling and splitting so the next measurement means
something. 7.4–7.6 decide what "means something" is *before* looking at the answer.

### 7.1 · Purge the split boundaries, and fix the metadata *(F11)*

**Files:** [`src/labels/dataset.py:14-57`](src/labels/dataset.py#L14-L57), [`scripts/build_features.py:42-59`](scripts/build_features.py#L42-L59)

Labels are computed on the full frame before splitting, so the last `horizon` rows of train embed
prices from val's first bars. Two rows per boundary at h=2 — negligible now, real the moment the
horizon grows. The walk-forward code already purges correctly; the plain split does not.

```python
def split_chronological(df, *, train_frac=0.70, val_frac=0.15, purge: int = 0):
    ...
    train = df.iloc[: n_train - purge]
    val = df.iloc[n_train : n_train + n_val - purge]
    test = df.iloc[n_train + n_val :]
```

`build_features.py` passes `purge=settings.labels.horizon`.

**Also fix the latent metadata bug in the same function** —
[`dataset.py:48`](src/labels/dataset.py#L48):

```python
        "first_ts_ms": {s: int(df["ts_ms"].iloc[0]) for s in SPLITS},
```

Every split reports the *global* first timestamp while the neighbouring `last_ts_ms` correctly uses
each partition. `metadata.json` is how a split gets audited after the fact, and for val and test it
currently shows something that would read as total overlap.

**Tests:** with `purge=h`, `train.ts_ms.max()` plus `h` intervals is strictly before
`val.ts_ms.min()`, same for val→test; `purge=0` reproduces today's behavior exactly; and
`first_ts_ms` differs across the three splits.

### 7.2 · Past-only label threshold *(F12)*

**File:** [`src/labels/labeler.py:33-37`](src/labels/labeler.py#L33-L37)

The threshold is a rolling std of *forward* returns whose window ends at t, so `fwd_return[t]`
helps set its own boundary. Not feature leakage — the threshold is not a model input — but it makes
labels self-referential and irreproducible at inference.

```python
fwd_return = close.pct_change(cfg.horizon).shift(-cfg.horizon)
# threshold from TRAILING realized volatility — fully observable at t
past_return = close.pct_change(cfg.horizon)
vol = past_return.rolling(cfg.threshold_window, min_periods=cfg.threshold_window // 2).std()
thr = (cfg.threshold_sigma * vol).clip(lower=cfg.min_abs_threshold)
```

**Tests:** a leakage probe in the style of the existing indicator probes — mutate `close` at
`t+1..t+h` and assert `label_threshold[t]` does not change.

### 7.3 · Make label changes invalidate models *(v1 8.5)*

**Files:** [`src/features/manifest.py`](src/features/manifest.py), [`src/models/store.py`](src/models/store.py), [`scripts/run_bot.py:63-74`](scripts/run_bot.py#L63-L74)

`feature_set_id` covers features only. 7.2 changes the *target* while leaving `fid` identical, so
the staleness guard — the mechanism that refuses to serve a model whose features have drifted —
will not notice. A model trained on old labels would be served silently against new ones.

Add `label_set_id = sha256(label_params)` alongside the feature manifest, store it in model
metadata, and check it in the same guard.

**Tests:** changing `labels.horizon` makes a stored model fail the guard.

### 7.4 · A holdout the selection never sees *(F10)*

**Files:** [`scripts/build_features.py`](scripts/build_features.py), [`scripts/train_model.py`](scripts/train_model.py), [`config/settings.yaml`](config/settings.yaml)

`settings.yaml` says it outright: the 60m/h=2 config is the *"only config with a real edge"* found
by scanning — and the gate then validates that survivor on the same test split the scan consulted.
PF 1.06 over 98 trades was never distinguishable from selection luck, and on clean data it is not
there at all.

Carve the most recent N months (suggest 4, ≈2,900 bars at 60m) into a `holdout` split that
`build_features.py` writes and **no training, tuning, or config scan may read**. The promotion gate
scores on `test`; the holdout is consulted exactly once, per candidate, and a candidate that has
been scored on it cannot be revised and re-scored.

Alternatively — and better if the config scan is going to continue — roll the evaluation window
forward per retrain so no fixed window is ever consulted twice.

**Tests:** `load_dataset` exposes the holdout separately; a guard in `train_model.py` raises if the
holdout frame is touched during fitting.

### 7.5 · Establish whether *any* config has positive gross edge

**Files:** [`scripts/benchmark_rulesets.py`](scripts/benchmark_rulesets.py), [`scripts/backtest.py`](scripts/backtest.py)

Before tuning thresholds or sizing, answer the prior question: does the signal have positive
expectancy *before costs* anywhere in the config space, on clean data, out of sample?

Report per candidate, on `test` only:
- gross P&L per trade, in basis points of notional, with a confidence interval
- the same after fees and modeled slippage (~15bp round trip at current settings)
- trade count, and the fraction of the equity curve's return attributable to the top 5 trades

A config whose *gross* per-trade edge is not distinguishable from zero is not a candidate,
regardless of what its net curve looks like on any single window. Record the whole scan, not just
the survivor — the record of how many configs were tried is what makes the survivor's p-value
interpretable.

**Done when:** there is a written answer to "how many configs were evaluated, and what is the best
gross per-trade edge with its interval" — even if the answer is "none clear zero".

### 7.6 · Pre-register the promotion criteria, then burn in

**File:** [`config/settings.yaml`](config/settings.yaml) `model.min_promote`

Write the pass criteria down *before* the run that tests them, in the config, in a commit that
precedes the result. Suggested shape, to be argued with rather than adopted blindly:

| Criterion | Threshold | Why |
| --- | --- | --- |
| Gross per-trade edge | > 20bp, CI excludes 0 | Must clear ~15bp round-trip costs with margin |
| Holdout trades | ≥ 100 | 98 was never enough to distinguish edge from noise |
| Holdout PF | ≥ 1.15 | 1.0 is break-even; 1.06 was inside noise |
| Max drawdown | ≤ 15% | Sizing sanity |
| Top-5-trade share | < 50% of return | Not one lucky week |

Then the real gate: **4–8 weeks of paper trading on live bars**, with a realized-vs-backtest report.
Divergence beyond modeled slippage means the fill assumptions are wrong, not that the edge is real.

---

### 🚦 Gate: edge, or no edge

A decision point, and both outcomes are legitimate:

- **Edge found** — a config clears the pre-registered criteria on an untouched holdout and survives
  burn-in. Proceed to Phase 8.
- **No edge** — nothing clears. Then the correct move is to stop building infrastructure and either
  change the hypothesis (different horizon, different instrument, non-directional strategies) or
  keep the bot as the well-engineered harness it is. **Do not proceed to Phases 8–9 on a strategy
  that failed this gate.** Phase 9 in particular is days of work whose only purpose is moving real
  money.

---

## Phase 8 — Operations *(F8, F9)*

**Effort:** ~1.5 days. Was v1 Phase 5.

Only worth doing once something is worth monitoring. Every failure mode currently ends as a log
line in `/tmp` and a dead process.

### 8.1 · Notifier

**File:** new `src/monitoring/notify.py`

Small interface, one implementation (Telegram bot or Discord webhook, ~30 lines), a no-op default
so paper and tests need no configuration. Called on: kill-switch trip, order failure,
reconciliation mismatch, daily-loss halt, start, and stop.

### 8.2 · Close the dead failure counter

**File:** [`scripts/run_bot.py:125-140`](scripts/run_bot.py#L125-L140)

`consecutive_failures` is incremented, logged, and never compared against anything. v1's 4.2 sketch
included `if consecutive_failures == 3: notifier.alert(...)`, which was dropped when Phase 4 landed
because the notifier did not exist yet — leaving literal dead code and no escalation path.

API failures do reach the kill switch through the gate, so *those* streaks work. Everything else —
the 5.2 snapshot bug, a full disk, a malformed feature row — loops forever at one attempt per bar.
Restore the alert, and add a hard ceiling:

```python
                if consecutive_failures == 3:
                    notifier.alert("repeated tick failures", str(exc))
                if consecutive_failures >= settings.risk.max_tick_failure_streak:
                    log.error("HALT — %d consecutive tick failures", consecutive_failures)
                    return 3
```

**Tests:** a runner stubbed to raise persistently exits 3 after the configured streak rather than
looping.

### 8.3 · Dead-man heartbeat

A healthchecks.io-style ping per tick, so *silence* alerts. This is what catches the failure modes
that kill the process outright.

### 8.4 · Rotating file logs

`logging.log_dir` is configured in [`config/settings.yaml:95`](config/settings.yaml#L95) and never
read. Wire `setup_logging` to a `RotatingFileHandler` under it.

### 8.5 · Supervised deployment *(F9)*

`run.sh` still uses `nohup`, a PID file in `/tmp`, and `LD_LIBRARY_PATH=$HOME/.local/lib`. Pick one
— a systemd unit inside WSL (`Restart=on-failure`, journald) or a Docker container with a restart
policy. Either **must** honor the F4 tombstone so auto-restart cannot resurrect a tripped bot.
Document the choice in the README.

**Done when:** kill the process mid-position and confirm it restarts, refuses to trade on a live
tombstone, and you are notified within a minute.

---

## Phase 9 — Close the live-execution gap *(F6, F7, F20, F21)*

**Effort:** ~3–4 days, then 2 weeks on testnet. Was v1 Phase 6. **Gated on Phase 7 passing.**

### 9.1 · Exchange-native stops and targets *(F6)*

Confirmed absent — no `set_trading_stop` call anywhere in the source. If the process dies holding a
position, nothing on Bybit protects it. And even alive, the bot reacts only at bar close: paper
books a stop fill at the stop price mid-bar, while live sends a market order after the bar closes
at whatever price then prevails. Live results will diverge from the paper ledger on every stopped
trade, always unfavorably.

Attach position-level TP/SL via Bybit V5 `set_trading_stop` on entry; treat the client-side check as
reconciliation, not enforcement.

### 9.2 · Adopt real fills, fees, and funding *(F7, F20)*

In testnet/live the broker still books fills at *simulated* prices and sizes off *simulated*
equity. `get_equity()` is implemented at
[`bybit_executor.py:146`](src/execution/bybit_executor.py#L146) and never called outside its own
test. After each order, fetch the actual execution by `orderLinkId` and book that — avg price, cum
fee, realized funding. Use exchange equity for sizing in live mode.

### 9.3 · Continuous reconciliation

Reconciliation runs once at startup and after an idempotent re-place. Re-reconcile position and
equity every bar, alert past a drift tolerance. 5.5 already made the comparison symmetric.

### 9.4 · Clock-drift check *(F21)*

`tick()` classifies bars as closed using `time.time()`; server time is checked only during warmup.
Compare against `server_time_ms()` periodically and alert past a threshold.

### 9.5 · Prove it on testnet

Two weeks minimum, `mode=testnet`, with: ≥10 round trips including a stop-out and a take-profit; a
deliberate mid-position `kill -9` → restart → confirm the exchange stop held and reconciliation
adopted the true state; a deliberate mismatch (close the position manually on the exchange) →
confirm trip, alert, tombstone, no auto-restart; and a realized-vs-expected fill report.

---

### 🚦 Gate: live money

Phase 7's burn-in passed on an untouched holdout, Phases 8–9 proven end-to-end on testnet, runbook
written and rehearsed. Keys are trade-only, no withdrawal permission, IP-whitelisted.

---

## Phase 10 — Hygiene and documentation *(F16–F22)*

**Effort:** ~1 day. Was v1 Phase 9, minus 9.1 which moved to 5.1.

- **F16 · Batch cache persistence.** [`_persist_bars`](src/runner/runner.py#L240) reloads and
  rewrites all 21,600 rows every hour. Append or partition by month.
- **F17 · Portability.** Replace `sys.path.insert(0, __file__.rsplit("/", 2)[0])` in all four
  scripts with `Path(__file__).resolve().parents[1]`. With 5.2, 5.3 and 5.7 done, the repo becomes
  genuinely runnable on native Windows; `.venv` is currently a WSL venv whose interpreter is a
  broken symlink from the host.
- **F18 · README and runbook.** The README is two lines. Write the architecture sketch, mode
  semantics, the safety model, and — most importantly — what to do when the kill switch trips, how
  to reset it, and how to roll back a model.
- **F19 · Stale docstring.** [`paper_broker.py:13-15`](src/execution/paper_broker.py#L13-L15)
  claims the backtester never arms cooldown; [`engine.py:170`](src/backtesting/engine.py#L170) arms
  it on stop-loss exits.
- **F22 · Secrets hygiene.** Still clean. Add a gitleaks pre-commit hook to keep it mechanical.
- **LightGBM on Windows.** Four tests fail with a native access violation on fit and on joblib
  round-trip, while LightGBM 4.7.0 fits fine standalone in the same environment. Triage under the
  Phase 6 Windows CI leg; the `test_store_roundtrip` failure is the interesting one given F13's
  warning about unpickling across library majors.

---

## Summary

| Phase | Scope | Effort | Blocks |
| --- | --- | --- | --- |
| 5 | Correctness, honesty, suite | ~1 day | Everything |
| 6 | CI | ~0.5 day | Safe refactoring in 7 |
| 7 | **Settle the science** | ~1 wk + 4–8 wk burn-in | **Whether 8–10 are worth doing** |
| 8 | Operations | ~1.5 days | Testnet |
| 9 | Live execution | ~4 days + 2 wk testnet | Live money |
| 10 | Hygiene, docs, runbook | ~1 day | — |

Phases 5 and 6 are worth doing regardless of how Phase 7 resolves: they fix real bugs and make the
test suite enforceable. Phases 8–10 are only worth their time if Phase 7 finds something.
