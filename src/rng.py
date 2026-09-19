"""Deterministic, per-process RNG utilities.

Training stochasticity comes from the process-global generator, which expman
seeds from ``experiment.seed`` and restores at every Stage boundary.  What is
left here are the two places that need to be independent of that stream: the
missing-mask generation, whose seed is its own ``missing_seed`` parameter, and
the per-step random bases, which must reproduce the same basis at a given
optimizer step without consuming the shared stream.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

MAX_SEED = 2**63 - 1


def derive_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps(
        [int(base_seed), *parts],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="big", signed=False) % MAX_SEED
    return seed if seed > 0 else 1


def numpy_rng(base_seed: int, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *parts))


@contextlib.contextmanager
def torch_seed_scope(base_seed: int, *parts: Any) -> Iterator[int]:
    # Some module constructors and third-party model loaders consume the global
    # Torch RNG, so fork_rng confines that global seeding to the current block
    # and restores the caller's state afterwards.
    seed = derive_seed(base_seed, *parts)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield seed


def configure_deterministic_backend() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
