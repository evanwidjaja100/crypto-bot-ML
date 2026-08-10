"""Run the event-driven backtester with the latest model on the test split.

Usage:
    python scripts/backtest.py [--dataset data/datasets/<id>] [--model <model_id>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.config import load_settings
from src.data_ingestion.intervals import INTERVAL_MS
from src.labels.dataset import load_dataset
from src.models.store import latest_model, load_model
from src.monitoring.logging_setup import setup_logging
from src.strategy.signal_engine import PositionState, decide

import logging

log = logging.getLogger("backtest")


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None, help="model_id; defaults to latest")
    parser.add_argument("--out", default="data/backtests")
    args = parser.parse_args(argv)

    setup_logging(settings.logging.log_level)

    artifacts_dir = Path("artifacts")
    if args.model:
        model, meta = load_model(args.model, artifacts_dir)
    else:
        loaded = latest_model(artifacts_dir)
        if loaded is None:
            log.error("no trained model found; run scripts/train_model.py first")
            return 1
        model, meta = loaded
    log.info("model: %s (framework=%s)", meta["model_id"], meta["framework"])

    if args.dataset:
        dataset_dir = args.dataset
    else:
        candidates = sorted(
            (Path(settings.data.data_dir) / "datasets").glob("*/"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not candidates:
            log.error("no datasets found; run scripts/build_features.py first")
            return 1
        dataset_dir = str(candidates[0])
    splits, ds_meta = load_dataset(dataset_dir)
    test = splits["test"]
    feature_cols = [c for c in test.columns if c.startswith("f_")]
    log.info("dataset=%s test_rows=%d", Path(dataset_dir).name, len(test))

    if meta["feature_set_id"] != ds_meta["feature_set_id"]:
        log.warning(
            "feature_set mismatch: model=%s dataset=%s — results are NOT valid; refusing.",
            meta["feature_set_id"], ds_meta["feature_set_id"],
        )
        return 2

    proba = model.predict_proba(test[feature_cols])
    proba_by_ts = {int(ts): p for ts, p in zip(test["ts_ms"].to_numpy(), proba)}

    def decision_fn(row: pd.Series, state: PositionState):
        return decide(
            row, state, settings.strategy, settings.risk,
            proba_by_ts[int(row["ts_ms"])],
        )

    engine = BacktestEngine(
        test,
        decision_fn,
        initial_equity=settings.backtest.initial_equity,
        taker_fee=settings.execution.taker_fee,
        maker_fee=settings.execution.maker_fee,
        slippage_bps=settings.execution.slippage_bps,
        funding_rate=settings.backtest.funding_rate,
        risk_cfg=settings.risk,
        interval_ms=INTERVAL_MS[settings.interval],
    )
    result = engine.run()
    metrics = result["metrics"]
    print(f"=== backtest {meta['model_id']} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")

    out_dir = Path(args.out) / meta["model_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result["equity"].to_parquet(out_dir / "equity.parquet", index=False)
    result["trades"].to_parquet(out_dir / "trades.parquet", index=False)
    result["decisions"].to_parquet(out_dir / "decisions.parquet", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"metrics": {k: (float(v) if isinstance(v, float) else v) for k, v in metrics.items()}}, indent=2)
    )
    print(f"saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
