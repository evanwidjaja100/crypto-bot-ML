# Implementation Checklist — crypto bot + ML

Status tracker for [`implementation-plan-v2.md`](implementation-plan-v2.md).
Source of truth for claiming a phase "done" is its **Done when** section and the
**Test** at commit time.

**Working rules (v2):** one phase per branch · one task per commit · test-first
for every behavioral fix · never delete data — quarantine it · bot stays in
**paper mode** through Phase 8; keys are first needed at Phase 9.

Legend: `[x]` done · `[/]` in progress · `[ ]` pending · `[!]` blocked/deferred.

---

## Phase 0–4 — Baseline (complete per changelog, not re-opened)
- [x] 0. Scoping + env conventions
- [x] 1. Data hygiene (mainnet-only data, network-keyed cache, candle validation — F1)
- [x] 2. Trading network derived from mode, reject legacy `BYBIT_TESTNET` (F2)
- [x] 3. Kill-switch tombstone + daily-loss persistence (F3, F4)
- [x] 4. Run-loop hardening (F5) + documented gate
- [x] Baseline: 153 pass / 9 fail (Windows) verified

## Phase 5 — Correctness, honesty, suite
- [x] 5.1 Retire losing model; explicit `active` (registry `{active, models}`, `active_model()`, migrate legacy list) — **F15**
- [x] 5.2 Atomic snapshots `os.replace` (5.3/8.4 pattern) — **Windows runner tests green**
- [x] 5.3 Explicit `encoding="utf-8"` on every text read/write
- [x] 5.4 `--once` reports failure in exit code (0 clean / 1 failure)
- [x] 5.5 Reconciliation symmetric (`abs(qty - want)`); under-hold trips
- [x] 5.6 Wire position-count to reality at both `approve_entry` call sites
- [x] 5.7 Suite collects under bare `pytest` (`pythonpath = ["."]`)
- [x] Pull-forward: F17 portable `sys.path` (6 scripts runnable on native Windows)
- [x] **Gate:** env rebuilt via `uv sync --extra dev`; `run_bot.py --once` exits 2; suite 166–170 green (only 4 known LightGBM-on-Windows fail)

## Phase 6 — CI (F14)
- [x] Ruff + mypy config in `pyproject.toml`
- [x] Fix mypy backlog (4 real type bugs in limits/engine/walk_forward/train)
- [x] `ruff check .`, `ruff format --check .`, `mypy src` all pass
- [x] `.github/workflows/ci.yml` (ubuntu+windows matrix, `uv sync --extra dev --frozen`)
- [x] README CI badge
- [!] LightGBM-on-Windows (4 tests) — **to triage on the Windows CI leg** (Phase 10)
- [x] `test_throttle_enforces_minimum_interval` — pre-existing flaky timing test, made deterministic

## Phase 7 — Settle the science (gate for everything below)
- [x] 7.1 Purge split boundaries (split_chronological `purge=horizon`) + fix `first_ts_ms` metadata (F11)
- [x] 7.2 Past-only label threshold (trailing realized vol) (F12)
- [x] 7.3 `label_set_id` stored in model meta + run_bot/backtest guard
- [x] 7.4 Holdout carve + `load_dataset`/guard (`assert_no_holdout_leak`) (F10)
- [x] 7.5 Establish whether *any* config has positive gross edge — **42 configs scanned; report in artifacts/phase7_5_gross_edge_report.md**
- [x] 7.6 Pre-register promotion criteria in `settings.yaml` (7 holdout/burn-in criteria)
- [x] **Gate:** edge found (on untouched holdout + burn-in) or **no-edge decision** — *honest no-edge decision reached; holdout untouched*

## Phase 8 — Operations
- [x] 8.1 Notifier (`src/monitoring/notify.py`, Discord/Telegram/Webhook support with safe no-op default)
- [x] 8.2 Close dead failure counter (alert @3, hard ceiling streak → exit 3)
- [x] 8.3 Dead-man heartbeat (healthchecks-style `src/monitoring/heartbeat.py`)
- [x] 8.4 Rotating file logs (`logging.log_dir` in `src/monitoring/logging_setup.py`)
- [x] 8.5 Supervised deployment (Dockerfile, docker-compose.yml, systemd service unit), honors tombstone (F9)

## Phase 9 — Live execution
- [x] 9.1 Exchange-native stops/targets (`set_trading_stop` / `cancel_trading_stop` attached to entries) (F6)
- [x] 9.2 Real fills/fees/funding (`get_equity`, `get_executions`, `get_closed_pnl`) (F7, F20)
- [x] 9.3 Continuous reconciliation (per-bar position & equity check with kill-switch trip & alerts)
- [x] 9.4 Clock-drift check (`check_clock_drift` against Bybit server time) (F21)
- [x] 9.5 Testnet integration test harness (`tests/test_live_execution.py`)
- [ ] **Gate:** live money (2-week testnet soak test completed, runbook rehearsed, keys trade-only/IP-whitelisted)

## Phase 10 — Hygiene & docs
- [x] F16 Batch cache persistence (`CandleStore.append` added and wired to `_persist_bars`)
- [x] F17 Portability (done early — unblocked 5.1 gate on Windows)
- [x] F18 README + runbook (architecture, modes, kill-switch reset, rollback runbook written)
- [x] F19 Stale docstring (`paper_broker.py` cooldown docstring corrected)
- [x] F22 Secrets hygiene (`.pre-commit-config.yaml` with gitleaks + ruff configured)
- [x] LightGBM-on-Windows triage (triaged & fixed via ctypes argtypes pointer safety)
