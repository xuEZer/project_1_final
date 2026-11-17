"""Run-local outputs and explicit checkpoint serialization."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from src.runtime import _plain

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def output_dir(ctx):
    # Stage artifacts are stored beside the run state.
    path = Path(ctx._store.root) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _git_metadata():
    if not (PROJECT_ROOT / ".git").exists():
        return None, None

    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env["GIT_OPTIONAL_LOCKS"] = "0"

    def git(*args):
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            env=env,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != PROJECT_ROOT:
            return None, None
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return None, None


def provenance(cfg):
    files = [
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "expman").rglob("*.py")),
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "pyproject.toml",
    ]
    sources = {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in files
    }
    config = _plain(cfg)
    git_commit, git_dirty = _git_metadata()
    return {
        "config": config,
        "source_sha256": sources,
        "source_digest": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        "config_digest": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "protocol": {
            "optimizer": "AdamW",
            "maximum_epochs": config["training"]["epochs"],
            "validation_patience": 5,
            "checkpoint_selection": "final",
            "forecast_context": "primary_mask",
            "forecast_mode": "evaluation",
            "imputer_initialization": "matched_random_seed",
        },
    }


def save_weights(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def archive_sources(folder):
    import tarfile

    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for name in ("src", "expman", "configs", "pyproject.toml", "uv.lock"):
            archive.add(
                PROJECT_ROOT / name,
                arcname=name,
                filter=lambda x: None if "__pycache__" in x.name else x,
            )
