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

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

from expman import Pipeline, Stage

from src.logger import setup_logger

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
    # Declared once: the old code reads these without a fallback, and setting
    # them inside the field loop would let a later scalar wipe an earlier value.
    parser.add_section(_SCALAR_SECTION)
    for blank, default in _REQUIRED_BLANK.items():
        parser.set(_SCALAR_SECTION, blank, default)
    for key, value in run.items():
        # expman freezes nested nodes into FrozenDict, which is a Mapping but
        # not a dict; testing for dict would silently flatten every section.
        if isinstance(value, Mapping):
            section = key
            fields = value
        else:
            section = _SCALAR_SECTION
            fields = {key: value}
        if not parser.has_section(section):
            parser.add_section(section)
        for field, field_value in fields.items():
            parser.set(
                section, _SCALAR_ALIAS.get(field, field), _scalar(field_value)
            )
    # expman forbids .get()/in on ctx.cfg, so every field read here must be
    # declared by the experiment YAML; a missing one is a loud KeyError.
    # expman freezes lists into FrozenList, which is a Sequence but not a list;
    # testing for list silently skipped this branch and left [training] device
    # unset, which the imputer builders read as config["training"]["device"].
    resources = run["device"]
    if isinstance(resources, Sequence) and not isinstance(resources, str):
        parser.set("training", "device", "cuda" if len(resources) else "cpu")
    dataset = run["dataset"]
    parser.set("data", "dataset", _scalar(dataset["name"]))
    for section, fields in runtime.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for field, field_value in fields.items():
            parser.set(section, field, _scalar(field_value))
    return parser


def _plain(value):
    """Unwrap expman's frozen containers into plain Python values."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_plain(item) for item in value]
    return value


def run_tag(run: dict) -> str:
    """Stable, filesystem-safe identifier for one run configuration.

    The readable prefix names the matrix cell; the digest suffix guarantees that
    configurations differing only in a sweep field (rq4 vs rq4_timing vary the
    epsilon and delta lists alone) do not share one artifact directory.
    """
    parts = [
        _scalar(run["stage"]),
        _scalar(run["dataset"]["name"]),
        _scalar(run["imputer"]["name"]),
        _scalar(run["predictor"]["name"]),
        f"seed{_scalar(run['seed'])}",
    ]
    payload = json.dumps(_plain(run), sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=4).hexdigest()
    return "_".join(parts) + f"__{digest}"


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


def artifact_dir(ctx) -> Path:
    """Per-run artifact directory, keyed by the run's own configuration."""
    root = Path(_scalar(ctx.cfg["paths"]["cache_dir"])) / "expman" / run_tag(ctx.cfg)
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_logger(ctx) -> logging.Logger:
    """One log file per run; appended on a retry so earlier attempts survive."""
    mode = "w" if getattr(ctx, "attempt", 1) <= 1 else "a"
    return setup_logger(
        artifact_dir(ctx) / "run.log",
        log_name=f"expman_{run_tag(ctx.cfg)}",
        mode=mode,
    )


def runtime_from_dataset(dataset: dict) -> dict:
    """Shape the loader discovers, which main() used to write back into cfg."""
    return {
        "d_feature": str(dataset["train"]["X_N"].shape[2]),
        "lookback": str(dataset["train"]["X_N"].shape[1]),
    }


class Train(Stage):
    """Build the imputer plus frozen forecaster, train it, and save the result.

    The model lives only inside this stage: expman snapshots every stage return
    value, so an ``nn.Module`` cannot cross the boundary.  What crosses is the
    path to the saved imputer state dict.
    """

    def process(self, data, ctx):
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from main import (
            build_dataloaders,
            configure_deterministic_runtime,
            init_joint_model,
        )
        from src.data_loader import load_h5
        from src.experiments import run_experiment

        dataset = load_h5(Path(data["cache_dir"]) / "dataset.h5")
        runtime = runtime_from_dataset(dataset)
        cfg = build_config(ctx.cfg, data=runtime)
        seed = cfg.getint("experiment", "seed")
        logger = make_logger(ctx)
        configure_deterministic_runtime(seed, logger)
        logger.info("expman train start: %s", run_tag(ctx.cfg))

        out = artifact_dir(ctx)
        writer = SummaryWriter(log_dir=out / "tb")
        try:
            model = init_joint_model(cfg, logger, writer)
            train_loader, eval_loader, no_gt = build_dataloaders(
                config=cfg, seed=seed, dataset=dataset,
                joint_model=model, logger=logger,
            )
            model = run_experiment(
                exp_type=cfg.get("experiment", "type"),
                joint_model=model,
                train_loader=train_loader,
                config=cfg,
                logger=logger,
            )
        finally:
            writer.close()

        checkpoint = out / "imputer.pt"
        torch.save(model.imputer.state_dict(), checkpoint)
        ctx.report_metric("train_done", 1)
        logger.info("expman train done, checkpoint=%s", checkpoint)
        return {
            "checkpoint": str(checkpoint),
            "cache_dir": data["cache_dir"],
            **runtime,
        }


class Evaluate(Stage):
    """Rebuild the imputer from the saved state dict and score every split."""

    SPLITS = ("train", "val", "test")

    def process(self, data, ctx):
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from main import (
            build_dataloaders,
            configure_deterministic_runtime,
            init_joint_model,
        )
        from src.data_loader import load_h5
        from src.evaluator import evaluate
        from src.rng import torch_seed_scope

        runtime = {"d_feature": data["d_feature"], "lookback": data["lookback"]}
        cfg = build_config(ctx.cfg, data=runtime)
        seed = cfg.getint("experiment", "seed")
        logger = make_logger(ctx)
        configure_deterministic_runtime(seed, logger)

        writer = SummaryWriter(log_dir=artifact_dir(ctx) / "tb_eval")
        try:
            model = init_joint_model(cfg, logger, writer)
        finally:
            writer.close()
        model.imputer.load_state_dict(
            torch.load(data["checkpoint"], map_location=cfg.get("training", "device"))
        )
        train_loader, eval_loader, _no_gt = build_dataloaders(
            config=cfg, seed=seed,
            dataset=load_h5(Path(data["cache_dir"]) / "dataset.h5"),
            joint_model=model, logger=logger,
        )
        loaders = {
            "train": train_loader.get("train_eval"),
            "val": train_loader.get("val"),
            "test": eval_loader,
        }
        results: dict[str, dict] = {}
        for split in self.SPLITS:
            loader = loaders[split]
            if loader is None:
                continue
            with torch_seed_scope(seed, f"{split}_evaluation"):
                metrics = evaluate(
                    cfg, model, loader, getattr(loader, "no_gt", False),
                    logger, split_name=split,
                )
            results[split] = metrics
            for name, value in metrics.items():
                if value is not None:
                    ctx.report_metric(f"{split}_{name}", float(value))
            logger.info("expman evaluate %s: %s", split, metrics)
        return results


def build_pipeline() -> Pipeline:
    return Pipeline([PrepareData, Train, Evaluate], name="impute-then-forecast")


__all__ = [
    "Evaluate",
    "PrepareData",
    "Train",
    "build_config",
    "build_pipeline",
    "run_tag",
    "runtime_from_dataset",
]
