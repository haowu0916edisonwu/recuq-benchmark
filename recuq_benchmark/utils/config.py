from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - used only before dependencies are installed.
    yaml = None


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        merged = _merge_two(merged, config)
    return merged


def _merge_two(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_two(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(yaml_path: str | Path) -> dict[str, Any]:
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        config = _safe_load_yaml(f.read())
    if "experiment" not in config:
        return config
    experiment = deepcopy(config["experiment"])
    base_dir = path.parent
    experiment["model"] = _load_relative(base_dir, experiment["model"])
    experiment["datasets"] = [_load_relative(base_dir, p) for p in experiment.get("datasets", [])]
    experiment["scenarios"] = [_load_relative(base_dir, p) for p in experiment.get("scenarios", [])]
    return {"experiment": experiment}


def _load_relative(base_dir: Path, value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    path = (base_dir / value).resolve()
    with path.open("r", encoding="utf-8") as f:
        return _safe_load_yaml(f.read())


def _safe_load_yaml(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _minimal_yaml_load(text)


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Small fallback parser for this repo's simple config files.

    It supports nested dictionaries, scalar values, inline lists, and block
    lists. Install PyYAML for general YAML support.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: dict[int, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            value = _parse_scalar(stripped[2:].strip())
            if not isinstance(parent, list):
                key = pending_key.get(stack[-1][0])
                if key is None:
                    raise ValueError(f"Cannot attach list item without pending key: {raw_line}")
                grandparent = stack[-2][1]
                grandparent[key] = []
                parent = grandparent[key]
                stack[-1] = (stack[-1][0], parent)
            parent.append(value)
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            parent[key] = _parse_scalar(raw_value)
        else:
            parent[key] = {}
            pending_key[indent] = key
            stack.append((indent, parent[key]))
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
