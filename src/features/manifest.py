"""Phase 4: feature manifest — stable IDs so models are never trained on mixed feature sets."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def feature_set_id(version: str, feature_cols: list[str], params: dict) -> str:
    """Content-addressed id: any change in version/columns/params invalidates caches."""
    payload = json.dumps(
        {"version": version, "features": sorted(feature_cols), "params": params},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_manifest(
    manifest_dir: str | Path,
    *,
    version: str,
    feature_cols: list[str],
    params: dict,
    symbol: str,
    interval: str,
) -> dict:
    """Persist a manifest and return it. Fails if the id already exists with
    different content (guards against accidentally mixing feature versions)."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fid = feature_set_id(version, feature_cols, params)
    manifest = {
        "feature_set_id": fid,
        "version": version,
        "symbol": symbol,
        "interval": interval,
        "feature_cols": sorted(feature_cols),
        "params": params,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = manifest_dir / f"{fid}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["feature_cols"] != manifest["feature_cols"] or existing["params"] != params:
            raise ValueError(
                f"manifest collision for {fid}: content differs. Refusing to overwrite."
            )
    path.write_text(json.dumps(manifest, indent=2))
    return manifest
