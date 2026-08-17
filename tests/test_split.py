"""Chronological split tests: strict ordering, no overlap, metadata."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import make_candles

from src.config import LabelSettings
from src.labels.dataset import (
    assert_no_holdout_leak,
    load_dataset,
    save_dataset,
    split_chronological,
)
from src.labels.labeler import add_labels


@pytest.fixture
def labeled():
    df = make_candles(1000, seed=21)
    return add_labels(df, LabelSettings()).dropna(subset=["label"])


def test_splits_are_chronological_and_non_overlapping(labeled):
    train, val, test, _ = split_chronological(labeled)
    assert int(train["ts_ms"].iloc[-1]) < int(val["ts_ms"].iloc[0])
    assert int(val["ts_ms"].iloc[-1]) < int(test["ts_ms"].iloc[0])
    ts = train["ts_ms"].tolist() + val["ts_ms"].tolist() + test["ts_ms"].tolist()
    assert len(set(ts)) == len(ts)  # no row appears twice


def test_row_counts_conserved(labeled):
    n = len(labeled)
    train, val, test, meta = split_chronological(labeled)
    assert len(train) + len(val) + len(test) == n
    assert meta["n_rows"] == {"train": len(train), "val": len(val), "test": len(test)}


def test_metadata_contains_class_distribution(labeled):
    _, _, _, meta = split_chronological(labeled)
    for s in ("train", "val", "test"):
        dist = meta["class_distribution"][s]
        assert set(dist) == {"short", "flat", "long"}
        assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_test_split_is_smallest(labeled):
    train, val, test, _ = split_chronological(labeled)
    assert len(test) < len(train)


# ---------------------------------------------------------------- 7.1
def test_purge_separates_boundaries_by_horizon():
    """With purge=h, h intervals of separation exist at each boundary so no
    label in a split embeds the next split's prices."""
    h = 2
    df = add_labels(make_candles(1000, seed=21), LabelSettings(horizon=h)).dropna(subset=["label"])
    train, val, test, _ = split_chronological(df, purge=h)
    iv = int(df["ts_ms"].iloc[1] - df["ts_ms"].iloc[0])
    assert train["ts_ms"].max() + h * iv < val["ts_ms"].min()
    assert val["ts_ms"].max() + h * iv < test["ts_ms"].min()


def test_purge_zero_reproduces_legacy():
    df = add_labels(make_candles(1000, seed=21), LabelSettings()).dropna(subset=["label"])
    t0, v0, te0, m0 = split_chronological(df, purge=0)
    t1, v1, te1, m1 = split_chronological(df)
    pd.testing.assert_frame_equal(t0, t1)
    pd.testing.assert_frame_equal(v0, v1)
    pd.testing.assert_frame_equal(te0, te1)
    assert m0["n_rows"] == m1["n_rows"]
    assert m0["first_ts_ms"] == m1["first_ts_ms"]


def test_first_ts_ms_differs_across_splits():
    df = add_labels(make_candles(1000, seed=21), LabelSettings()).dropna(subset=["label"])
    _, _, _, meta = split_chronological(df)
    firsts = [meta["first_ts_ms"][s] for s in ("train", "val", "test")]
    assert len(set(firsts)) == 3
    assert firsts[0] < firsts[1] < firsts[2]


def test_purge_too_large_raises():
    df = add_labels(make_candles(300, seed=21), LabelSettings()).dropna(subset=["label"])
    with pytest.raises(ValueError, match="purge too large"):
        split_chronological(df, purge=len(df))


# ---------------------------------------------------------------- 7.4
def test_load_dataset_exposes_holdout(tmp_path):
    df = pd.DataFrame({"ts_ms": range(100), "label": [1] * 100})
    splits = {
        "train": df.iloc[:40],
        "val": df.iloc[40:60],
        "test": df.iloc[60:80],
        "holdout": df.iloc[80:],
    }
    save_dataset(tmp_path, splits, {"n_rows": {k: len(v) for k, v in splits.items()}})
    loaded, _ = load_dataset(tmp_path)
    assert "holdout" in loaded
    assert len(loaded["holdout"]) == 20


def test_assert_no_holdout_leak_detects_overlap():
    train = pd.DataFrame({"ts_ms": [1, 2, 3]})
    holdout = pd.DataFrame({"ts_ms": [3, 4]})  # shares ts=3
    with pytest.raises(ValueError, match="holdout"):
        assert_no_holdout_leak(train, holdout)


def test_assert_no_holdout_leak_accepts_disjoint():
    train = pd.DataFrame({"ts_ms": [1, 2]})
    holdout = pd.DataFrame({"ts_ms": [3, 4]})
    assert_no_holdout_leak(train, holdout)  # no raise
