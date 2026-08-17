"""Phase 6: LightGBM training with early stopping on the validation split."""

from __future__ import annotations

import ctypes
import sys
from typing import Any

import lightgbm as lgb
from lightgbm import LGBMClassifier

from ..config import LgbmSettings

# On 64-bit Windows, ctypes defaults undeclared arguments to 32-bit int, which
# truncates 64-bit pointer addresses in LGBM_DatasetSetField and causes access violations.
if sys.platform == "win32" and hasattr(lgb, "basic") and hasattr(lgb.basic, "_LIB"):
    try:
        lgb.basic._LIB.LGBM_DatasetSetField.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lgb.basic._LIB.LGBM_DatasetSetField.restype = ctypes.c_int32
    except Exception:
        pass


def train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    cfg: LgbmSettings,
    *,
    seed: int = 42,
) -> LGBMClassifier:
    """3-class LightGBM (0=short, 1=flat, 2=long) with balanced class weights."""
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
    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )
    return model
