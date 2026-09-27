"""expman pipeline for the impute-then-forecast experiments.

Three Stage boundaries.  :class:`PrepareData` returns the prepared tensors,
:class:`Train` fits the imputer and returns its weights, and :class:`Evaluate`
scores training imputation and validation/test forecasting.

Training and scoring are separate Stages so each declares the configuration it
reads. B1 trains on reconstruction alone; PILOT branches query the frozen
forecaster to estimate local forecast-gradient coordinates. Evaluation runs the
forecaster on validation and test splits.

:class:`Evaluate` rebuilds the tensors from the configuration instead of
receiving them from :class:`PrepareData`.  expman snapshots every Stage's return
value, so passing the dataset through :class:`Train` would store a full copy of
it once per training group instead of once per dataset.  Preprocessing is a
fraction of a second against the cached raw series and the mask streams are
derived from ``missing_seed``, so the rebuild is deterministic and cheap.

The model code still expects a ``ConfigParser``; :func:`build_config`
materialises one run dictionary into that shape.  This bridge is transitional
and disappears once the internals read the run dictionary directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from collections.abc import Mapping, Sequence
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import numpy as np
import torch
from expman import Pipeline, Stage

# Limit per-worker CPU pools.  A batch can run many independent GPU workers;
# allowing every process to claim all host cores creates hundreds of runnable
# threads without improving accelerator throughput.
_CPU_THREADS = max(1, int(os.environ.get("FOTSI_CPU_THREADS", "2")))
torch.set_num_threads(_CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch permits setting the inter-op pool only before parallel work has
    # started.  Re-imports in an already initialised test process are harmless.
    pass

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


def prepare_dataset(run: dict) -> dict:
    """Build the windowed tensors for one run configuration.

    Shared by :class:`PrepareData` and :class:`Evaluate`.  Both the raw-series
    lookup and the mask streams are deterministic in the configuration -- the
    natural and artificial masks come from RNG streams derived from
    ``missing_seed`` -- so calling this twice for the same run returns identical
    tensors.
    """
    from src.build import parse_column_index, resolve_missing_seed
    from src.data_loader import preprocess_dataset

    cfg = build_config(run)
    dataset_cfg = run["dataset"]
    return preprocess_dataset(
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
        dataset = prepare_dataset(ctx.cfg)
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


class Evaluate(Stage):
    """Score train, validation and test with the trained imputer.

    The tensors are rebuilt here rather than received from :class:`PrepareData`,
    because expman snapshots every Stage's return value and a dataset crossing
    the :class:`Train` boundary would be stored once per training group instead
    of once per dataset.  Both the raw-series reader and the mask generator are
    deterministic in the configuration, so the rebuild reproduces exactly the
    tensors the training Stage consumed.
    """

    SPLITS = ("train", "val", "test")

    @classmethod
    def config_dependencies(cls, cfg):
        return {
            "dataset": True,
            "data": True,
            "missing_seed": True,
            "imputer": True,
            "predictor": True,
            "loader": True,
            "training": True,
            "device": True,
            "seed": True,
            "paths": {"cache_dir": True},
        }

    def process(self, trained, ctx):
        from src.build import (
            build_dataloaders,
            configure_deterministic_runtime,
            init_joint_model,
        )
        from src.evaluator import evaluate

        dataset = prepare_dataset(ctx.cfg)
        runtime = runtime_from_dataset(dataset)
        cfg = build_config(ctx.cfg, data=runtime)
        seed = cfg.getint("experiment", "seed")
        logger = configure_logging()
        configure_deterministic_runtime(seed, logger)
        logger.info("expman evaluate start: %s", run_tag(ctx.cfg))

        model = init_joint_model(cfg, logger)
        model.imputer.load_state_dict(trained["imputer"])
        train_loader, eval_loader, _no_gt = build_dataloaders(
            config=cfg, dataset=dataset, joint_model=model, logger=logger,
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
            metrics = evaluate(
                cfg, model, loader, getattr(loader, "no_gt", False),
                logger, split_name=split, run_predictor=(split != "train"),
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



def _has_protocol_field(run, field: str) -> bool:
    # expman's frozen configuration supports indexed reads, but not membership
    # tests or dict.get().
    try:
        run["protocol"][field]
    except KeyError:
        return False
    return True


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _capture(model, train_loader, history: list[torch.Tensor]) -> dict:
    generator = train_loader["train"].generator
    return {
        "imputer": _cpu_copy(model.imputer.state_dict()),
        "optimizer": _cpu_copy(model.optimizer.state_dict()),
        "global_epoch": model.global_epoch,
        "global_optim_step": model.global_optim_step,
        "history": _cpu_copy(history),
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "loader_rng": generator.get_state().clone() if generator else None,
    }


def _restore(model, train_loader, state: dict) -> None:
    model.imputer.load_state_dict(state["imputer"])
    model.optimizer.load_state_dict(state["optimizer"])
    model.global_epoch = int(state["global_epoch"])
    model.global_optim_step = int(state["global_optim_step"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    generator = train_loader["train"].generator
    if generator is not None and state["loader_rng"] is not None:
        generator.set_state(state["loader_rng"])


class Warmup(Stage):
    """Train reconstruction once and retain the best forecast checkpoint."""

    @classmethod
    def config_dependencies(cls, cfg):
        # Deliberately excludes `stage` and `pilot`: B1 and B2 share this output.
        protocol = {"history_size": True}
        if _has_protocol_field(cfg, "warmup_max_epochs"):
            protocol["warmup_max_epochs"] = True
        if _has_protocol_field(cfg, "patience"):
            protocol["patience"] = True
        return {
            "dataset": True,
            "data": True,
            "missing_seed": True,
            "imputer": True,
            "predictor": True,
            "loader": True,
            "training": True,
            "device": True,
            "seed": True,
            "paths": {"cache_dir": True},
            "protocol": protocol,
        }

    def process(self, dataset, ctx):
        from src.build import (
            build_dataloaders,
            configure_deterministic_runtime,
            init_joint_model,
        )

        cfg = build_config(ctx.cfg, data=runtime_from_dataset(dataset))
        logger = configure_logging()
        configure_deterministic_runtime(cfg.getint("experiment", "seed"), logger)
        logger.info("shared warm-up start: seed=%d", cfg.getint("experiment", "seed"))
        model = init_joint_model(cfg, logger)
        train_loader, _, _ = build_dataloaders(
            config=cfg, dataset=dataset, joint_model=model, logger=logger,
        )
        val_loader = train_loader["val"]
        max_epochs = cfg.getint("protocol", "warmup_max_epochs", fallback=None)
        history_size = cfg.getint("protocol", "history_size")
        if (max_epochs is not None and max_epochs < 1) or history_size < 0:
            raise ValueError("warmup_max_epochs must be >=1 and history_size >=0")
        if cfg.has_option("protocol", "patience"):
            return self._process_patience(
                model, train_loader, val_loader, ctx, logger,
                max_epochs=max_epochs,
                history_size=history_size,
                patience=cfg.getint("protocol", "patience"),
            )


class ContinueTraining(Stage):
    """Continue both branches from the same best warm-up checkpoint."""

    @classmethod
    def config_dependencies(cls, cfg):
        protocol = (
            {"patience": True, "continuation_epochs": True}
            if _has_protocol_field(cfg, "patience")
            else {"continuation_epochs": True}
        )
        declaration = {
            "stage": True,
            "dataset": True,
            "data": True,
            "missing_seed": True,
            "imputer": True,
            "predictor": True,
            "loader": True,
            "training": True,
            "device": True,
            "seed": True,
            "paths": {"cache_dir": True},
            "protocol": protocol,
        }
        if str(cfg["stage"]) != "B1":
            declaration["pilot"] = True
        return declaration

    def process(self, warmup_state, ctx):
        from src.build import (
            build_dataloaders,
            configure_deterministic_runtime,
            init_joint_model,
        )
        from src.experiments import _run_b2

        patience_mode = _has_protocol_field(ctx.cfg, "patience")
        branch = str(ctx.cfg["stage"])
        if patience_mode and branch not in {"B1", "B2", "B3", "B4", "B5", "B6", "B7"}:
            raise ValueError("patience protocol requires B1--B7")

        dataset = prepare_dataset(ctx.cfg)
        cfg = build_config(ctx.cfg, data=runtime_from_dataset(dataset))
        logger = configure_logging()
        configure_deterministic_runtime(cfg.getint("experiment", "seed"), logger)
        branch = cfg.get("experiment", "type")
        if patience_mode:
            continuation_epochs = cfg.getint("protocol", "continuation_epochs")
            if continuation_epochs < 1:
                raise ValueError("continuation_epochs must be >=1")
            if branch == "B6":
                # No imputation warm-up: train PILOT from the initial state for
                # the same number of applied epochs as the best-checkpoint arms.
                branch_state = warmup_state["initial_state"]
                branch_epochs = warmup_state["best_epoch"] + continuation_epochs
                trigger_epoch = None
                warmup_epoch = 0
            else:
                branch_state = warmup_state["best_state"]
                branch_epochs = continuation_epochs
                trigger_epoch = (
                    warmup_state["stop_epoch"]
                    if warmup_state["stopped_by_patience"] else None
                )
                warmup_epoch = warmup_state["stop_epoch"]
        else:
            continuation_epochs = cfg.getint("protocol", "continuation_epochs")
            if (
                branch not in {"B1", "B2", "B3", "B4", "B5", "B6", "B7"}
                or continuation_epochs < 1
            ):
                raise ValueError(
                    "staged protocol requires B1–B7 and continuation_epochs >=1"
                )
            branch_state = (
                warmup_state["initial_state"] if branch == "B6" else warmup_state
            )
            branch_epochs = (
                warmup_state["global_epoch"] + continuation_epochs
                if branch == "B6" else continuation_epochs
            )
            trigger_epoch = warmup_state["trigger_epoch"]
            warmup_epoch = warmup_state["global_epoch"]
        record_metric, flush_metrics = metric_sink(ctx)
        model = init_joint_model(cfg, logger, record_metric)
        train_loader, _, _ = build_dataloaders(
            config=cfg, dataset=dataset, joint_model=model, logger=logger,
        )
        _restore(model, train_loader, branch_state)
        cfg.set("training", "max_epochs", str(branch_epochs))
        logger.info(
            "continuation branch=%s start_epoch=%d trigger_epoch=%s next_epochs=%d",
            branch, model.global_epoch, trigger_epoch, branch_epochs,
        )

        def on_epoch_end(epoch, train_loss, trained_model):
            val_imp = trained_model.compute_val_loss(train_loader["val"])
            val_pred = trained_model.compute_val_prediction_loss(
                train_loader["val"],
                rng_namespace=(
                    "warmup_validation" if patience_mode else "staged_validation"
                ),
                rng_step=0,
            )
            ctx.log_metrics({"loss": {
                "train_loss": float(train_loss),
                "val_imp_loss": float(val_imp),
                "val_pred_loss": float(val_pred),
            }}, step=epoch)
            logger.info(
                "continuation epoch=%d branch=%s val_imp_loss=%.9g val_pred_loss=%.9g",
                epoch, branch, val_imp, val_pred,
            )

        if branch == "B1":
            model.fit(
                data_loader=train_loader,
                phase_name="b1_staged",
                num_epoch=branch_epochs,
                on_epoch_end=on_epoch_end,
            )
        else:
            history = [
                displacement.to(device=next(model.imputer.parameters()).device)
                for displacement in branch_state["history"]
            ]
            _run_b2(
                model, train_loader, cfg, logger,
                on_epoch_end=on_epoch_end,
                exp_type=branch,
                initial_history=history,
            )
        flush_metrics()
        return {
            "imputer": _cpu_copy(model.imputer.state_dict()),
            "warmup_epoch": warmup_epoch,
            "best_epoch": (
                warmup_state["best_epoch"] if patience_mode else None
            ),
            "trigger_epoch": trigger_epoch,
            "branch": branch,
            "start_epoch": branch_state["global_epoch"],
            "branch_epochs": branch_epochs,
        }


def build_pipeline() -> Pipeline:
    return Pipeline(
        [PrepareData, Warmup, ContinueTraining, Evaluate],
        name="impute-then-forecast-staged",
    )
