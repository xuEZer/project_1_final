"""Independent RNG utilities for reproducible experiments.

The project uses one user-facing ``experiment.random_seed`` as a base seed,
but each stochastic process derives its own named seed from that base. This
keeps model initialization, data shuffling, missing-mask generation, and TIP
random directions independent from one another.
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
    """Derive a deterministic integer seed from a base seed and namespace."""
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
    """Create an independent NumPy generator for a named random process."""
    return np.random.default_rng(derive_seed(base_seed, *parts))


def torch_generator(base_seed: int, *parts: Any, device: str = "cpu") -> torch.Generator:
    """Create an independent Torch generator for a named random process."""
    generator_device = device if str(device).startswith("cuda") else "cpu"
    try:
        generator = torch.Generator(device=generator_device)
    except RuntimeError:
        generator = torch.Generator()
    generator.manual_seed(derive_seed(base_seed, *parts))
    return generator


@contextlib.contextmanager
def torch_seed_scope(base_seed: int, *parts: Any) -> Iterator[int]:
    """Temporarily seed global Torch RNGs for APIs without generator args.

    Some module constructors and third-party model loaders use global Torch RNG
    internally. ``fork_rng`` confines that global seeding to the current block
    and restores the caller state afterwards.
    """
    seed = derive_seed(base_seed, *parts)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield seed


def configure_deterministic_backend() -> None:
    """Configure deterministic backend behavior without seeding global RNGs."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
