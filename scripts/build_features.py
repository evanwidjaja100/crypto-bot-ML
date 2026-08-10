"""Build features + labels from the local candle cache and save chronological splits.

Usage:
    python scripts/build_features.py [--symbol BTCUSDT] [--interval 5] [--out data/datasets]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from src.config import load_settings
from src.data_ingestion.candle_downloader import CandleStore
from src.features.manifest import save_manifest
from src.features.pipeline import build_feature_frame
from src.labels.dataset import split_chronological, save_dataset
from src.labels.labeler import add_labels
from src.monitoring.logging_setup import setup_logging

import logging

log = logging.getLogger("build_features")


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=settings.symbol)
    parser.add_argument("--interval", default=settings.interval)
    parser.add_argument("--out", default=str(Path(settings.data.data_dir) / "datasets"))
    args = parser.parse_args(argv)

    setup_logging(settings.logging.log_level)
    store = CandleStore(settings.data.data_dir)
    df = store.load(args.symbol, args.interval)
    if df is None or df.empty:
        log.error("no cached candles; run scripts/download_data.py first")
        return 1

    featured, feature_cols = build_feature_frame(df, settings.features)
    labeled = add_labels(featured, settings.labels)
    labeled = labeled.dropna(subset=["label"]).reset_index(drop=True)
    log.info(
        "features=%d rows=%d (warmup/tail dropped=%d)",
        len(feature_cols), len(labeled), len(df) - len(labeled),
    )

    manifest = save_manifest(
        Path(args.out) / "manifest",
        version=settings.features.version,
        feature_cols=feature_cols,
        params=settings.features.model_dump(),
        symbol=args.symbol,
        interval=args.interval,
    )

    train, val, test, split_meta = split_chronological(labeled)
    meta = {
        "symbol": args.symbol,
        "interval": args.interval,
        "feature_set_id": manifest["feature_set_id"],
        "feature_version": settings.features.version,
        "label_params": settings.labels.model_dump(),
        "class_mapping": {"0": "short", "1": "flat", "2": "long"},
        **split_meta,
    }
    out_dir = save_dataset(
        Path(args.out) / f"{args.symbol}_{args.interval}_{manifest['feature_set_id']}",
        {"train": train, "val": val, "test": test},
        meta,
    )
    print(f"dataset saved -> {out_dir}")
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}")
    print(f"class dist (train): {meta['class_distribution']['train']}")
    print(f"features ({len(feature_cols)}): {feature_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
