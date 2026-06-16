from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class ExpeditionStore:
    def __init__(self, base_dir: str = "expeditions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, expedition_data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        data = _sanitize(expedition_data)

        if not filename:
            exp_id = data.get("expedition_id") or f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            filename = f"{exp_id}.json"

        path = self.base_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def load(self, path: str | Path) -> Dict[str, Any]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)