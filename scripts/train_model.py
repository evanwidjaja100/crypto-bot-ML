"""Train baseline (logistic) + LightGBM with walk-forward validation.

Loads the most recent dataset from scripts/build_features.py, evaluates on the
held-out test split, saves artifacts, and prints the promotion verdict.

Usage:
    python scripts/train_model.py [--dataset data/datasets/<id>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.config import load_settings
from src.data_ingestion.intervals import INTERVAL_MS
from src.labels.dataset import load_dataset
from src.models.baseline import train_logistic
from src.models.evaluate import classification_metrics, simulate_trading
from src.models.store import make_model_id, save_model
from src.models.train import train_lgbm
from src.models.walk_forward import walk_forward
from src.monitoring.logging_setup import setup_logging
from src.strategy.signal_engine import PositionState, decide

import logging

log = logging.getLogger("train_model")


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="dataset dir; defaults to newest under data/datasets")
    parser.add_argument("--artifacts", default=str(Path("artifacts")))
    args = parser.parse_args(argv)

    setup_logging(settings.logging.log_level)
    datasets_root = Path(settings.data.data_dir) / "datasets"

    dataset_dir = args.dataset
    if dataset_dir is None:
        candidates = sorted(datasets_root.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            log.error("no datasets found; run scripts/build_features.py first")
            return 1
        dataset_dir = str(candidates[0])

    splits, meta = load_dataset(dataset_dir)
    feature_cols = [c for c in splits["train"].columns if c.startswith("f_")]
    log.info("dataset=%s rows=(train=%d val=%d test=%d) features=%d",
             Path(dataset_dir).name, *(len(splits[s]) for s in ("train", "val", "test")), len(feature_cols))

    X_tr, y_tr = splits["train"][feature_cols], splits["train"]["label"].astype(int)
    X_va, y_va = splits["val"][feature_cols], splits["val"]["label"].astype(int)
    X_te, y_te = splits["test"][feature_cols], splits["test"]["label"].astype(int)

    # ---- baseline: logistic -------------------------------------------------
    log.info("training logistic baseline...")
    logistic = train_logistic(X_tr, y_tr, seed=settings.model.seed)
    lr_te = classification_metrics(y_te, logistic.predict(X_te), logistic.predict_proba(X_te))
    log.info("logistic test: acc=%.4f f1=%.4f log_loss=%.4f", lr_te["accuracy"], lr_te["f1_macro"], lr_te["log_loss"])

    # ---- primary: LightGBM --------------------------------------------------
    log.info("training LightGBM (early stopping on val)...")
    lgbm = train_lgbm(X_tr, y_tr, X_va, y_va, settings.model.lgbm, seed=settings.model.seed)
    lgbm_te = classification_metrics(y_te, lgbm.predict(X_te), lgbm.predict_proba(X_te))
    log.info("lgbm test: acc=%.4f f1=%.4f log_loss=%.4f", lgbm_te["accuracy"], lgbm_te["f1_macro"], lgbm_te["log_loss"])

    # ---- walk-forward on train+val ------------------------------------------
    X_wf = pd.concat([X_tr, X_va], ignore_index=True)
    y_wf = pd.concat([y_tr, y_va], ignore_index=True)
    ts_wf = np.concatenate([splits["train"]["ts_ms"].to_numpy(), splits["val"]["ts_ms"].to_numpy()])

    def fit_predict(X_train, y_train, X_val, y_val):
        m = train_lgbm(X_train, y_train, X_val, y_val, settings.model.lgbm, seed=settings.model.seed)
        return m.predict(X_val), m.predict_proba(X_val)

    log.info("walk-forward (%d folds, purge=%d)...", settings.model.walk_forward.n_splits, settings.labels.horizon)
    wf = walk_forward(
        X_wf, y_wf,
        n_splits=settings.model.walk_forward.n_splits,
        min_train_rows=settings.model.walk_forward.min_train_rows,
        purge=settings.labels.horizon,
        fit_and_predict=fit_predict,
        ts=ts_wf,
    )
    log.info("walk-forward aggregate: %s", {k: round(v, 4) for k, v in wf["aggregate"].items() if k.startswith("mean_")})
    log.info("walk-forward: executed %d/%d folds", wf["n_folds_executed"], wf["expected_folds"])

    # ---- trading proxy (approximate; Phase 7 backtester supersedes) ----------
    proxy = simulate_trading(
        splits["test"], lgbm.predict(X_te),
        taker_fee=settings.execution.taker_fee,
        slippage=settings.execution.slippage_bps / 10_000,
        interval_ms=INTERVAL_MS[settings.interval],
    )
    log.info("trading proxy (test, approximate): %s", {k: round(v, 4) for k, v in proxy.items()})

    # ---- promotion gate: real backtester on the test split (supersedes proxy) --
    te = splits["test"]
    proba_te = lgbm.predict_proba(X_te)
    proba_by_ts = {int(ts): p for ts, p in zip(te["ts_ms"].to_numpy(), proba_te)}
    strat_cfg = settings.strategy

    def decide_with_proba(row, state: PositionState):
        return decide(row, state, strat_cfg, settings.risk, proba_by_ts[int(row["ts_ms"])])

    engine = BacktestEngine(
        te, decide_with_proba, initial_equity=settings.backtest.initial_equity,
        taker_fee=settings.execution.taker_fee, maker_fee=settings.execution.maker_fee,
        slippage_bps=settings.execution.slippage_bps,
        funding_rate=settings.backtest.funding_rate, risk_cfg=settings.risk,
        interval_ms=INTERVAL_MS[settings.interval],
    ).run()["metrics"]
    log.info("engine gate (test, real backtester): %s",
             {k: round(v, 4) if isinstance(v, float) else v
              for k, v in engine.items() if k in ("total_return", "profit_factor", "n_trades", "win_rate", "total_fees")})

    # ---- verdict first: a FAILING model must never reach artifacts/, because
    # latest_model() (used by run_bot.py) would pick it up and deploy it. -------
    fid = meta["feature_set_id"]
    model_id = make_model_id(meta["symbol"], meta["interval"], fid)

    threshold = settings.model.min_promote.get("test_accuracy", 0.34)
    min_pf = settings.model.min_promote.get("min_engine_pf", 1.0)
    min_trades = settings.model.min_promote.get("min_engine_trades", 30)
    class_ok = lgbm_te["accuracy"] >= threshold
    engine_ok = (
        engine["total_return"] > 0.0
        and engine["profit_factor"] >= min_pf
        and engine["n_trades"] >= min_trades
    )
    verdict = "PASS" if class_ok and engine_ok else "FAIL"
    if verdict == "FAIL":
        log.warning(
            "promotion verdict: FAIL | classification sanity=%s (acc %.4f vs min %.4f) | engine gate=%s "
            "(ret %.4f, pf %.4f, ntrades %d vs min %d) — model NOT saved; the previous model stays in "
            "place. Do NOT deploy this model to paper/testnet/live.",
            class_ok, lgbm_te["accuracy"], threshold, engine_ok,
            engine["total_return"], engine["profit_factor"], engine["n_trades"], min_trades,
        )
        return 2

    # ---- save artifacts (PASS only) --------------------------------------------
    base_meta = {
        "symbol": meta["symbol"],
        "interval": meta["interval"],
        "feature_set_id": fid,
        "feature_version": meta["feature_version"],
        "label_params": meta["label_params"],
        "class_mapping": meta["class_mapping"],
        "gate_verdict": verdict,
        "metrics": {
            "logistic_test": lr_te,
            "lgbm_test": lgbm_te,
            "walk_forward": wf["aggregate"],
            "engine_gate": {k: engine[k] for k in ("total_return", "profit_factor", "n_trades")},
        },
    }
    save_model(logistic, {**base_meta, "model_id": model_id, "model_type": "logistic"}, args.artifacts, framework="sklearn")
    lgbm_meta = save_model(lgbm, {**base_meta, "model_id": model_id, "model_type": "lgbm"}, args.artifacts, framework="lightgbm")
    log.warning(
        "promotion verdict: PASS | classification sanity=True (acc %.4f vs min %.4f) | engine gate=True "
        "(ret %.4f, pf %.4f, ntrades %d vs min %d) — model_id=%s saved; safe to deploy.",
        lgbm_te["accuracy"], threshold,
        engine["total_return"], engine["profit_factor"], engine["n_trades"], min_trades,
        lgbm_meta["model_id"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
