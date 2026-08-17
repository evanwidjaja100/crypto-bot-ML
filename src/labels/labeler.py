"""Phase 5: volatility-relative 3-class labels (short / flat / long).

Class mapping (fixed across the project):
    0 = short, 1 = flat, 2 = long
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LabelSettings

CLASS_SHORT = 0
CLASS_FLAT = 1
CLASS_LONG = 2


def add_labels(df: pd.DataFrame, cfg: LabelSettings) -> pd.DataFrame:
    """Add label, fwd_return and label_threshold columns.

    label[t] = LONG  if fwd_return[t] >  thr[t]
               SHORT if fwd_return[t] < -thr[t]
               FLAT  otherwise
    thr[t] = max(threshold_sigma * rolling_std(fwd_return, threshold_window),
                 min_abs_threshold)

    Rows in the last `horizon` positions get NaN labels (their forward returns
    are not yet observable) and must be dropped before training.
    """
    out = df.copy()
    close = out["close"]

    fwd_return = close.pct_change(cfg.horizon).shift(-cfg.horizon)
    # Threshold from TRAILING realized volatility — fully observable at t (F12).
    # The old rolling std of fwd_return let fwd_return[t] help set its own
    # boundary, making labels self-referential and irreproducible at inference.
    past_return = close.pct_change(cfg.horizon)
    vol = past_return.rolling(cfg.threshold_window, min_periods=cfg.threshold_window // 2).std()
    thr = (cfg.threshold_sigma * vol).clip(lower=cfg.min_abs_threshold)

    label = pd.Series(CLASS_FLAT, index=out.index, dtype="float64")
    label[fwd_return > thr] = CLASS_LONG
    label[fwd_return < -thr] = CLASS_SHORT
    label[fwd_return.isna()] = np.nan

    out["label"] = label
    out["fwd_return"] = fwd_return
    out["label_threshold"] = thr
    return out
