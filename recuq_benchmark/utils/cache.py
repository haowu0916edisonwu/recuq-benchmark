from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path
from typing import Any


class CacheManager:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{self._safe_key(key)}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self.cache_dir / f"{self._safe_key(key)}.pkl"
        with path.open("wb") as f:
            pickle.dump(value, f)

    def build_key(
        self,
        model_name: str,
        dataset: str,
        scenario: str,
        instance_id: str,
        variant_name: str,
        prompt_hash: str,
    ) -> str:
        return "::".join([model_name, dataset, scenario, instance_id, variant_name, prompt_hash])

    def prompt_hash(self, history_text: list[str], candidate_texts: list[str]) -> str:
        payload = "\n<History>\n".join(history_text) + "\n<Candidates>\n" + "\n".join(candidate_texts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _safe_key(self, key: str) -> str:
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        if len(key) <= 180:
            return key
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return f"{key[:150]}_{digest}"
