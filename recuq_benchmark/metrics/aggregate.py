from __future__ import annotations

from collections import defaultdict
from typing import Any


def aggregate_records(records: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(group_key) for group_key in group_keys)
        groups[key].append(record)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        row = {group_key: value for group_key, value in zip(group_keys, key)}
        row["n"] = len(items)
        metric_names = sorted({metric for item in items for metric in item.get("metrics", {})})
        for metric_name in metric_names:
            values = [float(item["metrics"][metric_name]) for item in items if metric_name in item["metrics"]]
            row[metric_name] = sum(values) / len(values) if values else 0.0
        if any("parse_success" in item for item in items):
            row["parse_success_rate"] = sum(1.0 for item in items if item.get("parse_success")) / len(items)
        rows.append(row)
    return rows
