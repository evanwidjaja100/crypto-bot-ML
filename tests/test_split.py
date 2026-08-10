"""Chronological split tests: strict ordering, no overlap, metadata."""
from __future__ import annotations

import pytest

from conftest import make_candles
from src.config import LabelSettings
from src.labels.dataset import split_chronological
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
