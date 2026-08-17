"""End-to-end model tests on synthetic data: train, evaluate, persist, load."""

from __future__ import annotations

import pytest
from conftest import make_candles

from src.config import FeatureSettings, LabelSettings, LgbmSettings
from src.features.pipeline import build_feature_frame
from src.labels.dataset import split_chronological
from src.labels.labeler import add_labels
from src.models.baseline import train_logistic
from src.models.evaluate import classification_metrics
from src.models.store import active_model, load_model, make_model_id, save_model
from src.models.train import train_lgbm


@pytest.fixture
def dataset():
    df = make_candles(3000, seed=42, vol=0.002)
    featured, cols = build_feature_frame(df, FeatureSettings())
    labeled = add_labels(featured, LabelSettings()).dropna(subset=["label"])
    train, val, test, _ = split_chronological(labeled)
    return train, val, test, cols


def test_logistic_trains_and_evals(dataset):
    train, val, test, cols = dataset
    model = train_logistic(train[cols], train["label"].astype(int), seed=1)
    m = classification_metrics(
        test["label"].astype(int), model.predict(test[cols]), model.predict_proba(test[cols])
    )
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["log_loss"] > 0.0


def test_lgbm_trains_with_early_stopping(dataset):
    train, val, test, cols = dataset
    cfg = LgbmSettings(n_estimators=50, early_stopping_rounds=10)
    model = train_lgbm(
        train[cols], train["label"].astype(int), val[cols], val["label"].astype(int), cfg, seed=1
    )
    m = classification_metrics(
        test["label"].astype(int), model.predict(test[cols]), model.predict_proba(test[cols])
    )
    assert 0.0 <= m["accuracy"] <= 1.0
    assert model.predict_proba(test[cols]).shape[1] == 3


def test_lgbm_better_than_random_sanity(dataset):
    train, val, test, cols = dataset
    cfg = LgbmSettings(n_estimators=50, early_stopping_rounds=10)
    model = train_lgbm(
        train[cols], train["label"].astype(int), val[cols], val["label"].astype(int), cfg, seed=1
    )
    m = classification_metrics(
        test["label"].astype(int), model.predict(test[cols]), model.predict_proba(test[cols])
    )
    assert m["accuracy"] > 0.33  # above random for 3 classes (sanity only)


def test_store_roundtrip(tmp_path, dataset):
    train, val, test, cols = dataset
    cfg = LgbmSettings(n_estimators=20, early_stopping_rounds=5)
    model = train_lgbm(
        train[cols], train["label"].astype(int), val[cols], val["label"].astype(int), cfg, seed=1
    )
    model_id = make_model_id("BTCUSDT", "5", "abc12345")
    save_model(
        model,
        {
            "model_id": model_id,
            "symbol": "BTCUSDT",
            "interval": "5",
            "feature_set_id": "abc12345",
            "metrics": {"test_accuracy": 0.5},
        },
        tmp_path,
        framework="lightgbm",
    )
    loaded, loaded_meta = load_model(model_id, tmp_path)
    assert loaded_meta["framework"] == "lightgbm"
    assert "created_at" in loaded_meta
    pred_a = model.predict(test[cols])
    pred_b = loaded.predict(test[cols])
    assert (pred_a == pred_b).all()

    loaded_active, _ = active_model(tmp_path)
    assert loaded_active is not None


def test_active_model_none_when_empty(tmp_path):
    assert active_model(tmp_path) is None


def test_same_model_id_different_types_do_not_collide(tmp_path, dataset):
    """Two trainers on the same base id (same timestamp prefix) must not
    overwrite each other's artifacts; the registry must resolve the right one."""
    train, val, test, cols = dataset
    y = train["label"].astype(int)
    lgb = train_lgbm(
        train[cols],
        y,
        val[cols],
        val["label"].astype(int),
        LgbmSettings(n_estimators=10, early_stopping_rounds=5),
        seed=1,
    )
    lin = train_logistic(train[cols], y, seed=2)

    model_id = "BTCUSDT_5_abc12345_same_stamp"  # identical id for both
    save_model(lin, {"model_id": model_id, "model_type": "logistic"}, tmp_path, framework="sklearn")
    save_model(
        lgb, {"model_id": model_id, "model_type": "lightgbm"}, tmp_path, framework="lightgbm"
    )

    assert (tmp_path / "models" / f"{model_id}-logistic.pkl").exists()
    assert (tmp_path / "models" / f"{model_id}-lightgbm.pkl").exists()

    import json

    reg = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert reg["active"] == model_id
    assert reg["models"] == [
        {"model_id": model_id, "model_type": "logistic"},
        {"model_id": model_id, "model_type": "lightgbm"},
    ]

    loaded, meta = load_model(model_id, tmp_path)
    assert meta["model_type"] == "lightgbm"  # most recent registry entry wins
    assert (loaded.predict(test[cols]) == lgb.predict(test[cols])).all()


# ---------------------------------------------------------------- 5.1
def test_active_model_none_when_latest_recorded_model_is_fail(tmp_path):
    """A registry whose most recent entry is a FAIL must report nothing
    deployable — run_bot must not fall back to it by list position."""
    import json

    reg = {
        "active": None,
        "models": [
            {"model_id": "m_good", "model_type": "lgbm", "gate_verdict": "PASS"},
            {"model_id": "m_bad", "model_type": "lgbm", "gate_verdict": "FAIL"},
        ],
    }
    (tmp_path / "models.json").write_text(json.dumps(reg), encoding="utf-8")
    assert active_model(tmp_path) is None


def test_active_model_migrates_legacy_list_registry(tmp_path):
    """A legacy list-shaped registry (with duplicated ids) loads without
    raising, is migrated in place to the new shape, dedupes history, and
    reports no active model."""
    import json

    legacy = ["a", "a", "b", "b", {"model_id": "c", "model_type": "lgbm"}]
    (tmp_path / "models.json").write_text(json.dumps(legacy), encoding="utf-8")

    assert active_model(tmp_path) is None  # nothing promoted in a legacy registry

    reg = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert reg["active"] is None
    assert [m["model_id"] for m in reg["models"]] == ["a", "b", "c"]


def test_active_model_returns_explicitly_promoted_model(tmp_path):
    """Saving a (gate-PASS) model makes it the explicitly active model;
    active_model never infers from list position."""
    save_model(
        {"x": 1}, {"model_id": "m1", "model_type": "logistic"}, tmp_path, framework="sklearn"
    )
    save_model(
        {"x": 2}, {"model_id": "m2", "model_type": "logistic"}, tmp_path, framework="sklearn"
    )

    model, meta = active_model(tmp_path)
    assert meta["model_id"] == "m2"  # the promoted one, whichever came last
    assert model == {"x": 2}

    # the earlier one is still loadable by id (kept as history)
    older, _ = load_model("m1", tmp_path)
    assert older == {"x": 1}
