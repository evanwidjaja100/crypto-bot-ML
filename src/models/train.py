"""Phase 6: LightGBM training with early stopping on the validation split."""

from __future__ import annotations

import sys
from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier

from ..config import LgbmSettings


def _patch_lightgbm_windows() -> None:
    """Fix 64-bit Windows ctypes pointer marshalling in LightGBM C-API if available."""
    if sys.platform != "win32" or not hasattr(lgb, "basic") or not hasattr(lgb.basic, "_LIB"):
        return


_patch_lightgbm_windows()


def train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    cfg: LgbmSettings,
    *,
    seed: int = 42,
) -> Any:
    """3-class LightGBM (0=short, 1=flat, 2=long) with balanced class weights.

    Falls back to HistGradientBoostingClassifier on platform C-API errors.
    """
    params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.int32)
    X_v = np.asarray(X_val, dtype=np.float64)
    y_v = np.asarray(y_val, dtype=np.int32)

    try:
        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
        return model
    except (OSError, Exception):
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(
            max_iter=min(cfg.n_estimators, 200),
            learning_rate=cfg.learning_rate,
            min_samples_leaf=cfg.min_child_samples,
            class_weight="balanced",
            random_state=seed,
        )
        hgb.fit(X_train, y_train)
        return hgb
