"""Feature pipeline tests, including the end-to-end lookahead probe."""

from __future__ import annotations

import pandas as pd

from src.config import FeatureSettings
from src.features.manifest import feature_set_id, label_set_id
from src.features.pipeline import FEATURE_PREFIX, build_feature_frame


def test_feature_columns_all_prefixed_and_clean(candles, feature_settings):
    out, cols = build_feature_frame(candles, feature_settings)
    assert cols
    assert all(c.startswith(FEATURE_PREFIX) for c in cols)
    assert out[cols].isna().sum().sum() == 0
    # warm-up rows dropped
    assert len(out) < len(candles)


def test_pipeline_deterministic(candles, feature_settings):
    a, cols = build_feature_frame(candles, feature_settings)
    b, _ = build_feature_frame(candles, feature_settings)
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_no_lookahead_features(candles, feature_settings):
    """Perturb row i; features at every row < i must be identical."""
    i = 250
    df2 = candles.copy()
    df2.loc[i, ["open", "high", "low", "close"]] *= 10.0
    df2.loc[i, "volume"] *= 10.0

    base, cols = build_feature_frame(candles, feature_settings)
    pert, _ = build_feature_frame(df2, feature_settings)

    merged = base[["ts_ms"] + cols].merge(pert[["ts_ms"] + cols], on="ts_ms", suffixes=("_a", "_b"))
    before = merged[merged["ts_ms"] < candles["ts_ms"].iloc[i]]
    for c in cols:
        pd.testing.assert_series_equal(
            before[f"{c}_a"], before[f"{c}_b"], check_names=False, obj=f"feature {c}"
        )


def test_feature_set_id_stable_and_sensitive():
    cols = ["f_a", "f_b"]
    params = FeatureSettings().model_dump()
    a = feature_set_id("v1", cols, params)
    b = feature_set_id("v1", cols, params)
    assert a == b
    changed = dict(params, rsi_period=21)
    assert a != feature_set_id("v1", cols, changed)


def test_label_set_id_sensitive_to_label_params():
    """7.3: a label change (e.g. horizon) must change the label-set id so a
    model trained on the old labels is invalidated."""
    from src.config import LabelSettings

    a = label_set_id(LabelSettings(horizon=2).model_dump())
    b = label_set_id(LabelSettings(horizon=5).model_dump())
    assert a == label_set_id(LabelSettings(horizon=2).model_dump())  # deterministic
    assert a != b  # target changed -> id changed


def test_time_features_present(candles, feature_settings):
    _, cols = build_feature_frame(candles, feature_settings)
    for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert f"f_{name}" in cols


def test_atr_feature_scale_invariant_and_raw_kept(candles, feature_settings):
    """f_atr_14 must not depend on the price level (a model feature), while the
    raw atr_raw_14 column stays in price units for stop anchoring."""
    scaled = candles.copy()
    for col in ("open", "high", "low", "close", "turnover"):
        scaled[col] *= 100.0  # 100x price level, same shape

    base, _ = build_feature_frame(candles, feature_settings)
    hi, _ = build_feature_frame(scaled, feature_settings)

    assert "atr_raw_14" in base.columns and "atr_raw_14" not in [
        c for c in base.columns if c.startswith(FEATURE_PREFIX)
    ]
    assert "f_atr_14" in base.columns

    merged = base[["ts_ms", "f_atr_14", "atr_raw_14"]].merge(
        hi[["ts_ms", "f_atr_14", "atr_raw_14"]], on="ts_ms", suffixes=("_a", "_b")
    )
    # normalized feature identical under pure price scaling
    pd.testing.assert_series_equal(merged["f_atr_14_a"], merged["f_atr_14_b"], check_names=False)
    # raw ATR scales with the price level (stays in price units)
    pd.testing.assert_series_equal(
        merged["atr_raw_14_b"], merged["atr_raw_14_a"] * 100.0, check_names=False
    )
