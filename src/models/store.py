"""Phase 6: model persistence with provenance metadata and a registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

REGISTRY = "models.json"


def make_model_id(symbol: str, interval: str, feature_set_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{interval}_{feature_set_id[:8]}_{stamp}"


def _artifact_stem(model_id: str, model_type: str | None) -> str:
    """Model artifacts are unique per (id, type) so logistic/lgbm never collide."""
    return f"{model_id}-{model_type}" if model_type else model_id


def save_model(
    model,
    meta: dict,
    artifacts_dir: str | Path,
    *,
    framework: str = "unknown",
) -> dict:
    """Persist model + metadata; append to registry. Returns the meta dict."""
    artifacts_dir = Path(artifacts_dir)
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    model_id = meta["model_id"]
    model_type = meta.get("model_type", "model")
    stem = _artifact_stem(model_id, model_type)

    path = artifacts_dir / "models" / f"{stem}.pkl"
    joblib.dump(model, path)

    meta = {
        "model_id": model_id,
        "model_type": model_type,
        "framework": framework,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    (artifacts_dir / "models" / f"{stem}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    registry_path = artifacts_dir / REGISTRY
    reg = _read_registry(artifacts_dir)
    reg["models"].append({"model_id": model_id, "model_type": model_type})
    # Explicit promotion: only gate-PASS models reach save_model (train_model.py
    # returns 2 without saving on FAIL), so the just-saved model is the one to
    # deploy. `active` is a real pointer — never inferred from list order.
    reg["active"] = model_id
    _write_registry(registry_path, reg)
    return meta


def _resolve_model_type(model_id: str, artifacts_dir: Path) -> str | None:
    """model_type for the most recent registry entry with this id (None for legacy string entries)."""
    reg = _read_registry(artifacts_dir)
    for entry in reversed(reg["models"]):
        if (entry if isinstance(entry, str) else entry.get("model_id")) == model_id:
            return None if isinstance(entry, str) else entry.get("model_type")
    return None


def load_model(model_id: str, artifacts_dir: str | Path) -> tuple[object, dict]:
    artifacts_dir = Path(artifacts_dir)
    stem = _artifact_stem(model_id, _resolve_model_type(model_id, artifacts_dir))
    model = joblib.load(artifacts_dir / "models" / f"{stem}.pkl")
    meta = json.loads((artifacts_dir / "models" / f"{stem}.json").read_text(encoding="utf-8"))
    return model, meta


def _write_registry(registry_path: Path, reg: dict) -> None:
    registry_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _read_registry(artifacts_dir: Path) -> dict:
    """Read and normalize the registry into {active, models}.

    Migrates the legacy list shape (bare string ids or {model_id, model_type}
    dicts written by the pre-v2 code) in place: history is deduped and kept,
    and `active` stays null because nothing was explicitly promoted.
    """
    registry_path = artifacts_dir / REGISTRY
    if not registry_path.exists():
        return {"active": None, "models": []}
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        models: list[dict] = []
        seen: set[str] = set()
        for entry in raw:
            mid = entry if isinstance(entry, str) else entry["model_id"]
            if mid in seen:
                continue
            seen.add(mid)
            item: dict = {"model_id": mid}
            if not isinstance(entry, str) and entry.get("model_type"):
                item["model_type"] = entry["model_type"]
            models.append(item)
        reg = {"active": None, "models": models}
        _write_registry(registry_path, reg)
        return reg
    return {"active": raw.get("active"), "models": raw.get("models", [])}


def active_model(artifacts_dir: str | Path) -> tuple[object, dict] | None:
    """The explicitly promoted model, or None when nothing is deployable.

    `active` is a real pointer set at promotion; it is never inferred from list
    position, so a failing retrain cannot leave a known-bad model deployed.
    """
    artifacts_dir = Path(artifacts_dir)
    reg = _read_registry(artifacts_dir)
    active_id = reg["active"]
    if not active_id:
        return None
    return load_model(active_id, artifacts_dir)
