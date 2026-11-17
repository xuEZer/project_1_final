"""Single expman entry point for every YAML experiment."""

import hashlib
from pathlib import Path

from expman import Pipeline
from src.stages.data import PrepareData
from src.stages.evaluate import Evaluate
from src.stages.summarize import Summarize
from src.stages.train import Train


class ExperimentPipeline(Pipeline):
    def signature(self):
        root = Path(__file__).resolve().parent.parent
        digest = hashlib.sha256()
        for p in [
            *sorted((root / "src").rglob("*.py")),
            *sorted((root / "expman").rglob("*.py")),
            root / "uv.lock",
            root / "pyproject.toml",
        ]:
            digest.update(p.relative_to(root).as_posix().encode())
            digest.update(p.read_bytes())
        signatures = super().signature()
        for item in signatures:
            item["implementation_sha256"] = digest.hexdigest()
        return signatures


def build_pipeline():
    return ExperimentPipeline([PrepareData, Train, Evaluate, Summarize], name="pilot")
