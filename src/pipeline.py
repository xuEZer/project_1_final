"""expman pipeline for the impute-then-forecast experiments.

Stage boundaries are drawn so that only picklable descriptions cross them: the
model object is built, trained and checkpointed inside :class:`Train`, and
:class:`Evaluate` rebuilds the imputer from the run configuration and the saved
state dict.  Nothing holds a live ``DataLoader`` or ``nn.Module`` across a
boundary, because expman snapshots every stage's return value.

The model code still expects a ``ConfigParser``; :func:`build_config`
materialises one run dictionary into that shape.  This bridge is transitional
and disappears once the internals read the run dictionary directly.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

from expman import Pipeline, Stage

ROOT = Path(__file__).resolve().parent.parent

# Sections the downstream code reads.  A scalar top-level field such as
# ``experiment`` or ``seed`` belongs to the experiment section.
_SCALAR_SECTION = "experiment"

# Top-level scalars whose ConfigParser name differs from the YAML field.
_SCALAR_ALIAS = {"stage": "type"}

# Keys the old code reads without a fallback, so they must exist even when a
# run does not set them.
_REQUIRED_BLANK = {"missing_seed": "", "load_ckpt": "", "name": ""}


def _scalar(value: object) -> str:
    return "" if value is None else str(value)


def build_config(run: dict, **runtime: dict) -> ConfigParser:
    """Materialise one expman run into the ConfigParser the model code takes.

    ``runtime`` carries values derived while running (for example the feature
    count and lookback the loader discovers), which used to be written back
    into the config by ``main()``.  ``ctx.cfg`` is read-only, so they arrive
    here explicitly instead.
    """
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    for key, value in run.items():
        if isinstance(value, dict):
            section = key
            fields = value
        else:
            section = _SCALAR_SECTION
            fields = {key: value}
        if not parser.has_section(section):
            parser.add_section(section)
        if section == _SCALAR_SECTION:
            for blank, default in _REQUIRED_BLANK.items():
                parser.set(section, blank, default)
        for field, field_value in fields.items():
            parser.set(
                section, _SCALAR_ALIAS.get(field, field), _scalar(field_value)
            )
    dataset = run.get("dataset")
    if isinstance(dataset, dict) and "name" in dataset:
        parser.set("data", "dataset", _scalar(dataset["name"]))
    for section, fields in runtime.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for field, field_value in fields.items():
            parser.set(section, field, _scalar(field_value))
    return parser


def run_tag(run: dict) -> str:
    """Stable, filesystem-safe identifier for one run configuration."""
    parts = [
        _scalar(run.get("stage")),
        _scalar(run["dataset"]["name"]),
        _scalar(run["imputer"]["name"]),
        _scalar(run["predictor"]["name"]),
        f"seed{_scalar(run.get('seed'))}",
    ]
    return "_".join(parts)


class PrepareData(Stage):
    """Preprocess the dataset and report where the tensors landed."""

    def process(self, data, ctx):
        from main import parse_column_index
        from src.data_loader import preprocess_dataset

        cfg = build_config(ctx.cfg)
        dataset_cfg = ctx.cfg["dataset"]
        folder = preprocess_dataset(
            source=dataset_cfg["source"],
            repo_name=dataset_cfg["repo_name"],
            dataset_name=dataset_cfg["dataset_name"],
            frequency=dataset_cfg["frequency"],
            item_id=dataset_cfg["item_id"],
            column_index=parse_column_index(_scalar(dataset_cfg["column_index"])),
            missing_rate=cfg.getfloat("data", "missing_rate"),
            artificial_missing_rate=cfg.getfloat("data", "artificial_missing_rate"),
            block_size=cfg.getint("data", "block_size"),
            lookback_len=cfg.getint("data", "lookback"),
            pred_len=cfg.getint("data", "pred_len"),
            val_ratio=cfg.getfloat("data", "val_ratio"),
            test_ratio=cfg.getfloat("data", "test_ratio"),
            random_seed=cfg.getint("experiment", "seed"),
            logger=logging.getLogger("expman.pipeline"),
            save_path=Path(cfg.get("paths", "cache_dir")) / "data",
        )
        ctx.report_metric("cache_ready", 1)
        return {"cache_dir": str(folder)}


class Train(Stage):
    """Build the imputer plus frozen forecaster, train, and save the result."""

    def process(self, data, ctx):
        raise NotImplementedError(
            "S2 in progress: this stage wires init_joint_model, "
            "build_dataloaders and run_experiment for both B1 and B2."
        )


class Evaluate(Stage):
    """Rebuild the imputer from the saved state dict and score every split."""

    def process(self, data, ctx):
        raise NotImplementedError("S2 in progress: wraps src.evaluator.evaluate.")


def build_pipeline() -> Pipeline:
    return Pipeline([PrepareData, Train, Evaluate], name="impute-then-forecast")


__all__ = [
    "Evaluate",
    "PrepareData",
    "Train",
    "build_config",
    "build_pipeline",
    "run_tag",
]
