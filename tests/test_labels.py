"""Label correctness: formula, tail NaN, class coverage, causality probe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.config import LabelSettings
from src.labels.labeler import CLASS_FLAT, CLASS_LONG, CLASS_SHORT, add_labels


def test_forward_return_formula():
    df = pd.DataFrame({"close": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0], "ts_ms": range(6)})
    cfg = LabelSettings(horizon=2)
    out = add_labels(df, cfg)
    assert out["fwd_return"].iloc[0] == pytest.approx(4.0 / 1.0 - 1.0)
    assert out["fwd_return"].iloc[1] == pytest.approx(8.0 / 2.0 - 1.0)
    assert np.isnan(out["fwd_return"].iloc[-1])


def test_tail_labels_are_nan(candles, label_settings):
    h = label_settings.horizon
    out = add_labels(candles, label_settings)
    assert out["label"].iloc[-h:].isna().all()
    assert out["label"].iloc[: -h].notna().all()


def test_all_classes_present_on_volatile_data():
    df = make_candles(2000, seed=11, vol=0.002)
    out = add_labels(df, LabelSettings())
    labeled = out.dropna(subset=["label"])
    counts = labeled["label"].value_counts()
    assert CLASS_SHORT in counts and CLASS_LONG in counts and CLASS_FLAT in counts
    for c in (CLASS_SHORT, CLASS_FLAT, CLASS_LONG):
        assert counts[c] > 0.02 * len(labeled)  # each class materially present


def test_label_threshold_floor():
    df = make_candles(500, seed=4, vol=1e-9)  # essentially flat market
    cfg = LabelSettings(threshold_window=100, min_abs_threshold=1e-3)
    out = add_labels(df, cfg)
    assert (out["label_threshold"].dropna() >= 1e-3).all()


def test_labels_use_only_future_close_of_exact_horizon():
    """Causality probe: perturb close[i]; labels may only change in rows
    whose threshold window or forward return contains row i."""
    cfg = LabelSettings(horizon=5, threshold_window=10)
    df = make_candles(500, seed=13)
    out = add_labels(df, cfg)

    i = 300
    df2 = df.copy()
    df2.loc[i, "close"] *= 1.5
    out2 = add_labels(df2, cfg)

    idx = out.index.to_numpy()
    unaffected = (idx < i - cfg.horizon) | (idx >= i + cfg.threshold_window)
    # fwd_return changes exactly at rows {i-horizon, i}
    fwd_same = pd.Series(
        np.isclose(out["fwd_return"], out2["fwd_return"], equal_nan=True), index=out.index
    )
    assert not bool(fwd_same.loc[i - cfg.horizon])
    assert not bool(fwd_same.loc[i])
    others = (idx != i - cfg.horizon) & (idx != i)
    assert bool(fwd_same.loc[others].all())
    # labels must be identical everywhere outside [i-horizon, i+window)
    pd.testing.assert_series_equal(
        out.loc[unaffected, "label"], out2.loc[unaffected, "label"], check_names=False
    )
