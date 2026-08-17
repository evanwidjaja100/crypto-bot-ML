"""Phase 5: chronological dataset splits and on-disk persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
HOLDOUT = "holdout"
CLASS_NAMES = {0: "short", 1: "flat", 2: "long"}


def split_chronological(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    purge: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Strict time-ordered split. Never shuffles. Guards against overlap.

    `purge` drops that many rows from the tail of train and val. Labels are
    computed on the full frame before splitting, so the last `purge` rows of
    train embed prices from val (and val's from test); purging them (F11)
    removes the boundary leakage.
    """
    if df.empty:
        raise ValueError("cannot split an empty frame")
    if not df["ts_ms"].is_monotonic_increasing:
        raise ValueError("ts_ms must be monotonically increasing before splitting")
    if purge < 0:
        raise ValueError("purge must be >= 0")

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if purge >= n_train or purge >= n_val:
        raise ValueError("purge too large relative to the split sizes")
    train = df.iloc[: n_train - purge].copy()
    val = df.iloc[n_train : n_train + n_val - purge].copy()
    test = df.iloc[n_train + n_val :].copy()

    assert int(train["ts_ms"].iloc[-1]) < int(val["ts_ms"].iloc[0])
    assert int(val["ts_ms"].iloc[-1]) < int(test["ts_ms"].iloc[0])

    def _dist(part: pd.DataFrame) -> dict[str, float]:
        if part.empty:
            return {}
        counts = part["label"].value_counts(dropna=False).sort_index()
        return {
            CLASS_NAMES[int(k)]: float(v / len(part))
            for k, v in counts.items()
            if k == k  # skip NaN keys
        }

    meta = {
        "n_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "first_ts_ms": {
            s: int(part["ts_ms"].iloc[0])
            for s, part in [("train", train), ("val", val), ("test", test)]
        },
        "last_ts_ms": {
            s: int(part["ts_ms"].iloc[-1])
            for s, part in [("train", train), ("val", val), ("test", test)]
        },
        "class_distribution": {
            s: _dist(part) for s, part in [("train", train), ("val", val), ("test", test)]
        },
    }
    return train, val, test, meta


def assert_no_holdout_leak(train: pd.DataFrame, holdout: pd.DataFrame | None) -> None:
    """Raise if a training frame overlaps the holdout's time range.

    The holdout is carved from the most recent bars and must never be read by
    training, tuning, or a config scan (F10). This is the guard that makes an
    (accidental or future) leak a hard failure instead of a silent one.
    """
    if holdout is None or holdout.empty:
        return
    first_ho = int(holdout["ts_ms"].iloc[0])
    if train["ts_ms"].ge(first_ho).any():
        raise ValueError(
            "holdout frames overlap the training split — rebuild the dataset "
            "with build_features.py; training must never touch the holdout"
        )


def save_dataset(
    out_dir: str | Path,
    splits: dict[str, pd.DataFrame],
    meta: dict,
) -> Path:
    """Write train/val/test parquet files plus metadata.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
    meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_dataset(dataset_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    dataset_dir = Path(dataset_dir)
    splits = {name: pd.read_parquet(dataset_dir / f"{name}.parquet") for name in SPLITS}
    holdout_path = dataset_dir / f"{HOLDOUT}.parquet"
    if holdout_path.exists():
        splits[HOLDOUT] = pd.read_parquet(holdout_path)
    meta = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    return splits, meta
