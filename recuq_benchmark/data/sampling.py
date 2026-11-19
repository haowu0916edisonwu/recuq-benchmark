from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return (int(base_seed) + int.from_bytes(digest, byteorder="big", signed=False)) % (2**32)


def deterministic_negative_sample(
    item_pool: Sequence[str],
    positives: set[str],
    excluded: set[str],
    num_negatives: int,
    seed: int,
    preferred_pool: Sequence[str] | None = None,
) -> list[str]:
    rng = random.Random(seed)
    selected: list[str] = []
    selected_set: set[str] = set()

    def add_from(pool: Sequence[str], remaining: int) -> None:
        if remaining <= 0 or not pool:
            return

        def maybe_add(item_id: str) -> None:
            if item_id in positives or item_id in excluded or item_id in selected_set:
                return
            selected.append(item_id)
            selected_set.add(item_id)

        max_attempts = max(remaining * 50, min(len(pool), 1000))
        for _ in range(max_attempts):
            maybe_add(pool[rng.randrange(len(pool))])
            if len(selected) >= num_negatives:
                return

        start = rng.randrange(len(pool))
        for offset in range(len(pool)):
            maybe_add(pool[(start + offset) % len(pool)])
            if len(selected) >= num_negatives:
                return

    if preferred_pool:
        add_from(preferred_pool, num_negatives)
    if len(selected) < num_negatives:
        add_from(item_pool, num_negatives - len(selected))
    return selected[:num_negatives]
