"""Phase 6: logistic regression baseline (sanity check, not the main model)."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic(X_train, y_train, *, seed: int = 42):
    """Standardized multinomial logistic regression with balanced classes."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
    ).fit(X_train, y_train)
