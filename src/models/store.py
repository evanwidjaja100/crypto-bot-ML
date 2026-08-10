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
    (artifacts_dir / "models" / f"{stem}.json").write_text(json.dumps(meta, indent=2))

    registry_path = artifacts_dir / REGISTRY
    entries = json.loads(registry_path.read_text()) if registry_path.exists() else []
    entries.append({"model_id": model_id, "model_type": model_type})
    registry_path.write_text(json.dumps(entries, indent=2))
    return meta


def _resolve_model_type(model_id: str, artifacts_dir: Path) -> str | None:
    """model_type for the most recent registry entry with this id (None for legacy string entries)."""
    registry_path = artifacts_dir / REGISTRY
    if not registry_path.exists():
        return None
    entries = json.loads(registry_path.read_text())
    for entry in reversed(entries):
        if (entry if isinstance(entry, str) else entry.get("model_id")) == model_id:
            return None if isinstance(entry, str) else entry.get("model_type")
    return None


def load_model(model_id: str, artifacts_dir: str | Path) -> tuple[object, dict]:
    artifacts_dir = Path(artifacts_dir)
    stem = _artifact_stem(model_id, _resolve_model_type(model_id, artifacts_dir))
    model = joblib.load(artifacts_dir / "models" / f"{stem}.pkl")
    meta = json.loads((artifacts_dir / "models" / f"{stem}.json").read_text())
    return model, meta


def latest_model(artifacts_dir: str | Path) -> tuple[object, dict] | None:
    """Most recently created model, or None."""
    artifacts_dir = Path(artifacts_dir)
    registry_path = artifacts_dir / REGISTRY
    if not registry_path.exists():
        return None
    entries = json.loads(registry_path.read_text())
    if not entries:
        return None
    last = entries[-1]
    return load_model(last if isinstance(last, str) else last["model_id"], artifacts_dir)
