# Production Readiness Review

**Repo:** `evanwidjaja100/crypto-bot-ML` @ `da8a835` · **Reviewed:** 2026-08-12
**Environment:** WSL Python 3.14.4, pandas 3.0.5, numpy 2.5.1, scikit-learn 1.9.0, LightGBM 4.7.0
**Test suite at time of review:** 133 passing

---

## Verdict

**Strong engineering skeleton, not yet safe to fund.**

The architecture is genuinely good — idempotent order placement, a single risk-approval gate, backtest/paper fill equivalence, leakage-probed features, and a promotion gate in front of deployment. Most hobby bots have none of this.

But one disqualifying data bug poisons everything downstream: the candle cache mixes Bybit **testnet** prices into mainnet history, so the promoted model, its PASS verdict, and the current paper session are all trained and evaluated on partly fictional prices. Alongside it: two risk-control state bugs that silently disarm the daily-loss limit and kill switch on restart, and a live-execution layer whose stops exist only client-side.

Fix the criticals before trusting any result; fix the highs before real keys touch mainnet.

| Severity | Count | Meaning |
| --- | --- | --- |
| Critical | 4 | Invalidates results or bypasses safety |
| High | 5 | Required before real keys touch mainnet |
| Medium | 5 | ML validity and reproducibility |
| Low | 8 | Hygiene and polish |

---

## Critical — invalidates results or bypasses safety

These four make current results untrustworthy and current safety rails weaker than they look. All are cheap to fix relative to what they protect.

### F1 · Candle cache is corrupted: testnet prices mixed into mainnet history

`data/raw/BTCUSDT_60.parquet` holds closes from **$50.80 to $1,999,999.80**; 2,467 of 19,321 bars (12.8%) jump more than 10% in one hour, spread across the entire 2024-05 → 2026-08 window (a 254% hourly jump as recently as June 2026). Mainnet BTC does not do this — Bybit testnet's thin books do.

**Root cause:** the runner builds its market-data client from the env flag — `BybitClient(testnet=settings.env.bybit_testnet)` in [`scripts/run_bot.py:53`](../scripts/run_bot.py#L53), and `.env` sets `BYBIT_TESTNET=true` — while [`scripts/download_data.py`](../scripts/download_data.py) defaults to mainnet. Both write the same cache file with no record of which network the rows came from, so warmup backfills and incremental updates interleave two different price universes.

**Blast radius:** the promoted model (`BTCUSDT_60_37ec52b2`), its PASS verdict (PF 1.06, +1.4% on test), every backtest, and the running paper session are all fit and scored on partly fictional prices.

**Fix:** market data should come from **mainnet always** — klines are public and need no keys; only *order* endpoints should ever point at testnet. Key the cache by network (or stamp a `network` field and refuse mixed writes), and extend `validate_candles` in [`src/data_ingestion/validation.py`](../src/data_ingestion/validation.py) with a max bar-to-bar move check (e.g. reject >25%/hour) so this class of corruption can never be written silently again. Then wipe the cache, redownload, rebuild features, retrain, and re-run the gate.

### F2 · Order network is decoupled from mode — the live-confirmation gate can be bypassed

The executor's session is built from `HTTP(testnet=settings.env.bybit_testnet, ...)` independent of `mode` ([`scripts/run_bot.py:79`](../scripts/run_bot.py#L79)). Set `mode=testnet` with `BYBIT_TESTNET=false` (say, forgotten after a data experiment) and the bot signs **real mainnet orders** while every log line says testnet — without ever asking for the `ENABLE-LIVE` phrase. The confirmation gate only guards the *mode string*, not the network that money actually moves on.

**Fix:** derive the trading network from mode — `testnet` mode ⇒ testnet endpoints, `live` mode ⇒ mainnet + confirm phrase — and make the `Settings` validator in [`src/config.py`](../src/config.py) reject inconsistent combinations outright. The env flag should disappear as an independent degree of freedom.

### F3 · Daily-loss limit forgets today's losses on restart

The snapshot saves `daily_loss_pnl`, but `restore_day_pnl()` restores only the number — not the day key or equity base. After a restart `_day` is `None`, so `allowed()` returns `True` unconditionally, and the first position close resets `_pnl` to zero before adding the new fill ([`src/risk/limits.py:58-77`](../src/risk/limits.py#L58-L77)). Net effect: lose 2%, crash (or get supervisor-restarted), and the bot resumes trading the same day with a clean slate. The restore path is currently dead weight.

**Fix:** persist and restore all three fields — day key, day P&L, and equity base — and have `allowed()` honor them immediately after restore. Add a test: trip the limit, snapshot, restore, assert entries are still rejected.

### F4 · Kill switch state is memory-only — any restart silently resets it

`KillSwitch` is designed to halt "until operator reset", but a tripped switch lives only in the process ([`src/risk/limits.py:7-42`](../src/risk/limits.py#L7-L42)). The runner exits with code 3 on a trip; the moment anything restarts the process (you, cron, or the supervisor you'll add for F9), the switch is forgotten — including trips caused by **reconciliation mismatch**, the one that means "the exchange and my ledger disagree about your money".

**Fix:** persist trips to a tombstone file (reason + timestamp) checked during startup; refuse to start while it exists. Reset becomes an explicit operator action: delete the file or run a `reset-kill-switch` script. Wire it into the snapshot tests.

---

## High — required before real keys touch mainnet

Paper mode tolerates these. Live mode does not.

### F5 · One hard API failure kills the whole process — the error-streak design never engages

`BybitClient` wraps any exhausted retry in `RuntimeError`; the run loop catches `RuntimeError` and exits with code 3 ([`scripts/run_bot.py:117-121`](../scripts/run_bot.py#L117-L121)). So a single network blip that outlasts the client's internal retries terminates the bot — the `max_api_error_streak=5` kill-switch logic can never accumulate a streak because the process dies on error #1. The first `tick()` at line 110 isn't covered by that handler at all, so it crashes with a raw traceback.

**Fix:** in the loop, distinguish kill-switch trips (exit) from transient failures (log, sleep one interval, continue) — e.g. a dedicated `KillSwitchTripped` exception type instead of matching on `RuntimeError`. Let the streak counter do the job it was built for.

### F6 · Stops and take-profits exist only client-side, evaluated once per closed bar

If the process dies holding a position, **nothing on the exchange protects it** — no stop order exists there. And even alive, the bot only reacts at bar close: paper mode books a stop fill at the stop price mid-bar ([`src/execution/paper_broker.py:204-229`](../src/execution/paper_broker.py#L204-L229)), but live the market order goes out after the bar closes at whatever price then prevails. Live results will systematically diverge from the paper ledger on every stopped trade, always in the unfavorable direction.

**Fix:** on entry in testnet/live, attach exchange-native TP/SL via Bybit V5 `set_trading_stop` (position-level) and treat the client-side check as reconciliation, not enforcement. This also makes crash-with-position survivable.

### F7 · Live ledger drifts from the exchange: simulated fills, simulated equity, one-shot reconciliation

In testnet/live the broker still books fills at *simulated* prices (bar open ± assumed slippage) and sizes positions off the *simulated* equity — actual fill price, actual fees, and `get_equity()` (implemented in [`src/execution/bybit_executor.py:141`](../src/execution/bybit_executor.py#L141) but never called) are ignored. Reconciliation runs once at startup and compares position side/qty only ([`src/runner/runner.py:361`](../src/runner/runner.py#L361)). Every real-world deviation (partial fill, fee tier, funding difference) accumulates silently into sizing and daily-loss math.

**Fix:** after each order, fetch the actual execution (avg price, cum fee, realized funding) by `orderLinkId` and book *that* into the ledger. Re-reconcile position and equity on a schedule (each bar is fine), alert past a drift tolerance, and use exchange equity for sizing in live mode.

### F8 · No alerting, no heartbeat — failure modes are silent

Kill-switch trip, reconciliation mismatch, order failure, daily-loss halt, or a plain crash all end as a log line in `/tmp` and a dead process. You find out when you next look. For a system whose whole job is unattended operation, "notice within minutes" is a core feature, not polish.

Also: `logging.log_dir` exists in [`config/settings.yaml`](../config/settings.yaml) but is never used — logs go to stdout only, unrotated, redirected to `/tmp` which vanishes on reboot.

**Fix:** a tiny notifier (Telegram bot or Discord webhook, ~30 lines) called on trip, order failure, reconciliation mismatch, daily-loss halt, and start/stop. Add a dead-man heartbeat (healthchecks.io-style ping per tick) so silence itself alerts. Write rotating file logs to the configured `log_dir`.

### F9 · No supervised deployment — nohup + PID file in /tmp

[`run.sh`](../run.sh) backgrounds the bot with `nohup`, tracks it via `/tmp/opencode/run_paper.pid`, and logs to `/tmp` — all erased on reboot, no restart on crash, no start-on-boot. The scripts also hardcode environment-specific paths (`$HOME/.local/lib`, WSL-only layout) that won't survive a machine move.

**Fix:** pick one — a systemd unit inside WSL (`Restart=on-failure`, `WantedBy=multi-user.target`, journald logging) or a Docker container with a restart policy. Either must honor the F4 tombstone so auto-restart can't resurrect a tripped bot. Document the choice in the README.

---

## Medium — ML validity and reproducibility

### F10 · The promotion gate measures a config that was selected on the same data

[`config/settings.yaml`](../config/settings.yaml) says it plainly: the 60m/h=2 config is the *"only config with a real edge"* found by scanning — and the gate then validates that survivor on the same test split the scan consulted. PF 1.06 over 98 trades is comfortably inside multiple-comparisons noise; the gate as constituted can't tell edge from selection luck. Retraining repeatedly against one fixed test window compounds this.

**Fix:** keep a final holdout the selection process never sees (most recent N months), or roll the evaluation window forward per retrain. Treat a mandatory paper burn-in — e.g. 4–8 weeks with realized-vs-backtest tracking — as the real promotion gate, with pre-registered pass criteria. On clean data (F1) the current metrics need re-establishing anyway.

### F11 · Chronological split leaks label horizon across boundaries

Labels are computed on the full frame before splitting, so the last `horizon` rows of train embed prices from val's first bars (and val→test likewise) — see [`src/labels/dataset.py:14`](../src/labels/dataset.py#L14) and [`scripts/build_features.py:42-59`](../scripts/build_features.py#L42-L59). At h=2 that's 2 rows per boundary — negligible today, but it becomes real if the horizon grows. The walk-forward code already purges correctly; the plain split doesn't.

**Fix:** drop `horizon` rows at each boundary in `split_chronological`, mirroring the walk-forward purge.

### F12 · Label threshold window includes the outcome it labels

The volatility that sets the label boundary is a rolling std of *forward* returns whose window ends at t — so `fwd_return[t]` participates in its own threshold ([`src/labels/labeler.py:33-37`](../src/labels/labeler.py#L33-L37)). Not feature leakage (the threshold isn't a model input), but it makes labels self-referential and irreproducible at inference time.

**Fix:** compute the threshold from past-only realized volatility (shift the window, or use trailing returns). Relabel and retrain alongside F1's rebuild.

### F13 · No dependency lock — the runtime is bleeding-edge and unpinned

[`pyproject.toml`](../pyproject.toml) declares open ranges (`pandas>=2.0`); the venv actually runs pandas 3.0.5, numpy 2.5.1, scikit-learn 1.9.0, Python 3.14.4. Any reinstall may produce a different (or broken) environment, and saved models may not unpickle across sklearn/joblib majors. Two warnings already point at breakage-in-waiting: a numpy deprecation inside joblib, and an overflow in `annualized_return` for degenerate equity curves ([`src/models/evaluate.py:93`](../src/models/evaluate.py#L93)).

**Fix:** adopt `uv` (or pip-tools) and commit the lock file; recreate the venv from it. Fix the overflow (clamp the exponent or guard short curves) and record model artifacts' library versions in their metadata.

### F14 · No CI — 133 good tests that only run when someone remembers

The test suite is the project's best asset (leakage probes, engine/paper equivalence, executor idempotency) and it's invisible to the GitHub repo it's pushed to. There's also no lint/format/typecheck configuration despite fully type-annotated code.

**Fix:** GitHub Actions running `ruff check` + `ruff format --check` + `mypy` + `pytest` on push/PR, against the locked environment from F13. Half a day, permanent payoff.

---

## Low — hygiene and polish

### F15 · Model registry: implicit "last entry wins", mixed formats, no provenance

`artifacts/models.json` mixes legacy string entries with dicts, holds duplicates, and "active model" is defined as whatever was appended last ([`src/models/store.py:76`](../src/models/store.py#L76)) — deployment by side effect. Add an explicit `active` pointer, record the git commit and library versions in model metadata, and prune the superseded 5m-era artifacts.

### F16 · Per-bar cache persistence rewrites the whole parquet file

`_persist_bars` ([`src/runner/runner.py:210`](../src/runner/runner.py#L210)) loads and rewrites the full year+ of candles every hour. Harmless at this scale, but it grows linearly forever; batch appends or partition by month when convenient.

### F17 · Portability traps for the WSL setup

`sys.path.insert(0, __file__.rsplit("/", 2)[0])` (all four scripts) breaks on native Windows paths — use `Path(__file__).resolve().parents[1]`. [`watch.sh`](../watch.sh) reads `/proc` and regex-parses YAML. Fine as WSL dev tools; document the WSL requirement so future-you doesn't debug it cold.

### F18 · Empty README, 400 KB session log in the repo, no runbook

The README is two lines; the design rationale lives in a committed AI-session transcript (`session-ses_0250.md`). Write the real thing: architecture sketch, mode semantics, safety model, and a runbook — what to do when the kill switch trips, how to reset, how to roll back a model. Move the session log to `docs/` or out of the repo.

### F19 · Stale docstring contradicts the backtester

[`src/execution/paper_broker.py:13-15`](../src/execution/paper_broker.py#L13-L15) claims the backtester never arms cooldown; [`src/backtesting/engine.py:170`](../src/backtesting/engine.py#L170) arms it on stop-loss exits. The code agrees — the comment doesn't. Fix the doc before it misleads a future change.

### F20 · Funding is modeled as a constant; live funding varies and flips sign

±0.01%/8h constant is fine for backtests (state the assumption), but live should book the *actual* funding payments from the exchange (part of F7's fill adoption) so the ledger and daily-loss math stay honest.

### F21 · Closed-bar detection trusts the local clock

`tick()` classifies bars as closed using `time.time()` ([`src/runner/runner.py:166`](../src/runner/runner.py#L166)); server time is checked only during warmup. A skewed clock quietly shifts decision timing. Compare against `server_time_ms()` periodically and alert past a threshold.

### F22 · Forward-looking secrets hygiene

Nothing is leaked today — verified: `.env` untracked, no key material in the session log or git history. Keep it that way mechanically: a gitleaks/pre-commit hook, and when live keys are created — trade-only permissions, no withdrawal, IP-whitelisted, documented in the runbook.

---

## What's already production-grade — keep these patterns

- **Idempotent order placement** — unique `orderLinkId` per attempt, order-history verification before any retry ([`src/execution/bybit_executor.py:81`](../src/execution/bybit_executor.py#L81)). Textbook handling of the one API call where blind retry loses money.
- **Backtest/paper equivalence as a tested invariant** — same fill rules, same conservative assumptions (stop touch fills, TP needs a close beyond, gap-through fills at the open), enforced by [`tests/test_engine_paper_equivalence.py`](../tests/test_engine_paper_equivalence.py).
- **Single risk-approval gate** in front of every order ([`src/risk/gate.py`](../src/risk/gate.py)), with reasons journaled for every rejection.
- **Causal feature discipline** — indicators tested mechanically for lookahead ([`tests/test_indicators.py`](../tests/test_indicators.py)); content-addressed feature manifests; the runner refuses to serve a model whose `feature_set_id` mismatches the live pipeline.
- **Promotion gating** — failing models are never written to artifacts ([`scripts/train_model.py:141`](../scripts/train_model.py#L141)), so the deployed-model pointer can't regress by accident.
- **Fail-fast typed config** — pydantic validation, explicit mode allowlist, live mode gated behind a confirmation phrase.
- **Crash-conscious persistence** — atomic snapshot writes (tmp + rename), per-bar journaling, gap backfill on restart so stops and funding are evaluated for every missed bar.
- **133 passing tests, ~2,400 lines** — covering indicators, labels, splits, walk-forward, broker, engine, executor, runner, and risk.

---

## Suggested order of work

Each step unblocks the next; 1–4 are small diffs, 5–7 are the real lift before live.

1. **Restore data integrity.** Mainnet-only market data, network-keyed cache, price-jump validator; wipe, redownload, rebuild, retrain, re-gate. *(F1)*
2. **Couple network to mode** and reject inconsistent config at startup. *(F2)*
3. **Make risk state survive restarts** — full daily-loss snapshot and a kill-switch tombstone, with tests. *(F3, F4)*
4. **Fix the crash-on-first-error loop** so the streak-based kill switch actually governs. *(F5)*
5. **Stand up operations** — alerts + heartbeat + rotating file logs, then a supervised deployment (systemd-in-WSL or Docker) that honors the tombstone. *(F8, F9)*
6. **Close the live-execution gap** — exchange-native TP/SL on entry, adopt real fills/fees/funding into the ledger, periodic position + equity reconciliation. Prove it on testnet. *(F6, F7, F20)*
7. **Lock and automate** — dependency lock file, CI with lint/typecheck/tests. *(F13, F14)*
8. **Tighten the science** — untouched holdout or rolling windows, boundary purge, past-only label thresholds, pre-registered paper burn-in criteria. *(F10, F11, F12)*
9. **Hygiene pass** — README + runbook, registry cleanup, docstring fix, clock drift check, pre-commit secrets scan. *(F15–F22)*

### Gates

| Gate | Requires |
| --- | --- |
| Keep paper-trading | Steps 1–4. Until then, paper results aren't measuring anything real. |
| Testnet with keys | Steps 5–6 proven end-to-end: alerts fire, restarts are safe, exchange stops exist. |
| Live money | Step 8's burn-in passed on clean data, with the runbook written and rehearsed. |
