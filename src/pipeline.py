"""expman pipeline for the impute-then-forecast experiments.

Two Stage boundaries.  :class:`PrepareData` returns the prepared tensors, and
:class:`TrainEvaluate` trains the imputer and scores every split in one process.
The dataset therefore crosses exactly one boundary: expman snapshots every
Stage's return value, so a third Stage would store this dataset again for every
run, and a separate scoring Stage would rebuild the model and reload the data
only because it runs in another process.

The model code still expects a ``ConfigParser``; :func:`build_config`
materialises one run dictionary into that shape.  This bridge is transitional
and disappears once the internals read the run dictionary directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
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
    """Preprocess the dataset and return the tensors.

    Nothing is written to disk: expman snapshots this Stage's return value, and
    that snapshot is what later runs reuse.  The declaration below names exactly
    the paths this reads, so two runs whose values match share one realization
    of the data -- the B1 and B2 branches, and every imputer/predictor pair, all
    see the same tensors.  It reads nothing else, in particular not ``stage``,
    the models or the training settings, which is what lets the scheduler run it
    once per dataset instead of once per run.
    """

    @classmethod
    def config_dependencies(cls, cfg):
        return {
            "dataset": True,
            "data": True,
            "paths": {"cache_dir": True},
            "missing_seed": True,
        }

    def process(self, data, ctx):
        from src.build import parse_column_index, resolve_missing_seed
        from src.data_loader import preprocess_dataset

        cfg = build_config(ctx.cfg)
        dataset_cfg = ctx.cfg["dataset"]
        dataset = preprocess_dataset(
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
            random_seed=resolve_missing_seed(cfg),
            logger=logging.getLogger("expman.pipeline"),
        )
        ctx.log_metrics({"data_ready": 1})
        return dataset


def configure_logging() -> logging.Logger:
    """Send this process's log records to stdout, which expman captures.

    Every attempt runs in its own worker whose stdout and stderr expman redirects
    into ``attempts/stage-<n>/<attempt>/output.log``; a second log file written by
    this code would only duplicate that.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger("expman.pipeline")


def metric_sink(ctx):
    """Forward a model's scalar diagnostics into expman's metric store.

    Returns ``(record, flush)``.  ``record(name, value, step)`` nests the
    slash-separated name, and the buffered rows are written when the step
    changes and by ``flush`` at the end.  expman pays a transaction per call, so
    reporting each of a step's scalars separately would spend more time in
    SQLite than in the optimizer.  Non-finite samples -- a ratio whose
    denominator was empty -- are dropped rather than refused.
    """
    pending: dict = {"step": None, "rows": {}}

    def flush() -> None:
        if pending["rows"]:
            ctx.log_metrics(pending["rows"], step=pending["step"])
            pending["rows"] = {}

    def record(name: str, value: float, step: int) -> None:
        number = float(value)
        if not math.isfinite(number):
            return
        if step != pending["step"]:
            flush()
            pending["step"] = step
        node = pending["rows"]
        parts = name.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = number

    return record, flush


def runtime_from_dataset(dataset: dict) -> dict:
    """Shape the loader discovers, which main() used to write back into cfg."""
    return {
        "d_feature": str(dataset["train"]["X_N"].shape[2]),
        "lookback": str(dataset["train"]["X_N"].shape[1]),
    }


class TrainEvaluate(Stage):
    """Train the imputer plus frozen forecaster, then score every split.

    Both halves live in one process because the prepared tensors arrive as this
    Stage's input.  Scoring in a separate Stage would mean returning them from
    here so the next Stage could read them: expman snapshots every return value,
    so that dataset would be stored once per run instead of once per dataset --
    and the scoring half would rebuild the model and reload the data purely
    because it runs in another process.

    No dependency declaration: both branches read the predictor.  Even B1 only
    trains the imputer, but its early stopping and best-checkpoint selection
    follow the validation *prediction* loss, so which imputer state is kept
    depends on the predictor.  That makes the branch-specific declaration expman
    documents (``B1`` names the imputer, ``B2`` adds the predictor) collapse to
    the whole configuration here, which is what the default already declares.
    Keep the default rather than listing the fields: a later field that this
    Stage reads would otherwise be missing from the list and silently reuse
    another run's result.
    """

    SPLITS = ("train", "val", "test")

    def process(self, dataset, ctx):
        from src.build import (
            build_dataloaders,
            configure_deterministic_runtime,
            init_joint_model,
        )
        from src.evaluator import evaluate
        from src.experiments import run_experiment

        runtime = runtime_from_dataset(dataset)
        cfg = build_config(ctx.cfg, data=runtime)
        seed = cfg.getint("experiment", "seed")
        logger = configure_logging()
        configure_deterministic_runtime(seed, logger)
        logger.info("expman train start: %s", run_tag(ctx.cfg))

        record_metric, flush_metrics = metric_sink(ctx)
        model = init_joint_model(cfg, logger, record_metric)
        if "imputer" in ctx.state:
            # Resuming: expman restored the state written by the last
            # successful checkpoint.save, so training continues rather
            # than restarting from the initial weights.
            model.imputer.load_state_dict(ctx.state["imputer"])
            logger.info(
                "resumed imputer state saved at epoch %s",
                ctx.state.get("next_epoch"),
            )
        train_loader, eval_loader, _no_gt = build_dataloaders(
            config=cfg, dataset=dataset, joint_model=model, logger=logger,
        )

        def on_epoch(epoch, monitor):
            """Record each epoch in expman's metric store and snapshot."""
            metrics = {
                key: float(value)
                for key, value in monitor.last_metrics.items()
                if value is not None and math.isfinite(float(value))
            }
            if metrics:
                ctx.log_metrics({"loss": metrics}, step=epoch)
            ctx.state["imputer"] = {
                key: value.detach().cpu()
                for key, value in model.imputer.state_dict().items()
            }
            ctx.state["next_epoch"] = epoch
            try:
                ctx.checkpoint.save(step=epoch)
            except RuntimeError:
                # --checkpoint requires a stage driven by Batch or
                # Experiment; a bare Pipeline.run has no checkpoint store.
                pass

        model = run_experiment(
            exp_type=cfg.get("experiment", "type"),
            joint_model=model,
            train_loader=train_loader,
            config=cfg,
            logger=logger,
            on_epoch=on_epoch,
        )
        flush_metrics()
        logger.info("expman train done")

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
            metrics = evaluate(
                cfg, model, loader, getattr(loader, "no_gt", False),
                logger, split_name=split,
            )
            results[split] = metrics
            measured = {
                name: float(value)
                for name, value in metrics.items()
                if value is not None and math.isfinite(float(value))
            }
            if measured:
                ctx.log_metrics({split: measured})
            logger.info("expman evaluate %s: %s", split, metrics)

        ctx.log_metrics({"train_done": 1})
        return results


def build_pipeline() -> Pipeline:
    return Pipeline(
        [PrepareData, TrainEvaluate], name="impute-then-forecast"
    )


__all__ = [
    "PrepareData",
    "TrainEvaluate",
    "build_config",
    "build_pipeline",
    "run_tag",
    "runtime_from_dataset",
]
