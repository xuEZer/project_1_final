"""Shared reconstruction warm-up followed by staged experiment branches.

The main protocol stops warm-up after ``patience`` consecutive epochs without
a new minimum forecast validation loss. B1 and B2 restore the same best
complete checkpoint and each train for ``continuation_epochs``.
Legacy continuation settings remain available for the existing ablation matrix.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

import numpy as np
import torch
from expman import Pipeline, Stage

from src.pipeline import (
    Evaluate,
    PrepareData,
    build_config,
    configure_logging,
    metric_sink,
    runtime_from_dataset,
)


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
        # Deliberately excludes `stage` and `tip`: B1 and B2 share this output.
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
        if max_epochs is None:
            raise ValueError("legacy warm-up requires warmup_max_epochs")
        history: list[torch.Tensor] = []

        def record_update(displacement: torch.Tensor) -> None:
            if history_size:
                history.append(displacement.detach().cpu().clone())
                del history[:-history_size]

        previous_loss = None
        previous_state = None
        initial_state = ctx.state.get("warmup_initial_snapshot")
        if "warmup_snapshot" in ctx.state:
            previous_state = ctx.state["warmup_snapshot"]
            previous_loss = float(ctx.state["warmup_pred_loss"])
            history = list(previous_state["history"])
            _restore(model, train_loader, previous_state)
            logger.info("warm-up resumed at epoch=%d", model.global_epoch)
        else:
            # Epoch zero is a real comparison point: a rise after the first
            # training epoch must branch from the initial checkpoint.
            previous_loss = model.compute_val_prediction_loss(
                val_loader, rng_namespace="warmup_validation", rng_step=0,
            )
            if not math.isfinite(previous_loss):
                raise RuntimeError("non-finite initial validation forecast loss")
            previous_state = _capture(model, train_loader, history)
            initial_state = previous_state
            ctx.state["warmup_initial_snapshot"] = initial_state
            ctx.log_metrics({"warmup": {"val_pred_loss": previous_loss}}, step=0)
            logger.info("warm-up epoch=0 val_pred_loss=%.9g", previous_loss)
            ctx.state["warmup_snapshot"] = previous_state
            ctx.state["warmup_pred_loss"] = previous_loss
            try:
                ctx.checkpoint.save(step=0)
            except RuntimeError:
                pass

        while model.global_epoch < max_epochs:
            model.fit(
                data_loader=train_loader,
                phase_name="shared_warmup",
                num_epoch=1,
                applied_update_observer=record_update,
            )
            epoch = model.global_epoch
            pred_loss = model.compute_val_prediction_loss(
                val_loader,
                rng_namespace="warmup_validation",
                rng_step=0,
            )
            if not math.isfinite(pred_loss):
                raise RuntimeError(f"non-finite warm-up validation forecast loss at epoch {epoch}")
            ctx.log_metrics({"warmup": {"val_pred_loss": pred_loss}}, step=epoch)
            logger.info("warm-up epoch=%d val_pred_loss=%.9g", epoch, pred_loss)
            if previous_loss is not None and pred_loss > previous_loss:
                logger.info(
                    "forecast loss rose %.9g -> %.9g; branching from epoch %d",
                    previous_loss, pred_loss, previous_state["global_epoch"],
                )
                return {
                    **previous_state,
                    "trigger_epoch": epoch,
                    "trigger_val_pred_loss": pred_loss,
                    "triggered": True,
                    "initial_state": initial_state,
                }
            previous_loss = pred_loss
            previous_state = _capture(model, train_loader, history)
            ctx.state["warmup_snapshot"] = previous_state
            ctx.state["warmup_pred_loss"] = previous_loss
            try:
                ctx.checkpoint.save(step=epoch)
            except RuntimeError:
                pass

        logger.info("no forecast-loss rise through epoch %d", max_epochs)
        return {
            **previous_state,
            "trigger_epoch": None,
            "trigger_val_pred_loss": None,
            "triggered": False,
            "initial_state": initial_state,
        }


    @staticmethod
    def _process_patience(
        model, train_loader, val_loader, ctx, logger, *,
        max_epochs: int | None, history_size: int, patience: int,
    ) -> dict:
        if patience < 1:
            raise ValueError("patience must be >=1")
        history: list[torch.Tensor] = []

        def record_update(displacement: torch.Tensor) -> None:
            if history_size:
                history.append(displacement.detach().cpu().clone())
                del history[:-history_size]

        initial_state = ctx.state.get("warmup_initial_snapshot")
        if "warmup_snapshot" in ctx.state:
            current_state = ctx.state["warmup_snapshot"]
            current_loss = float(ctx.state["warmup_pred_loss"])
            best_state = ctx.state["warmup_best_snapshot"]
            best_loss = float(ctx.state["warmup_best_pred_loss"])
            stale_epochs = int(ctx.state["warmup_stale_epochs"])
            history = list(current_state["history"])
            _restore(model, train_loader, current_state)
            logger.info(
                "warm-up resumed at epoch=%d best_epoch=%d stale_epochs=%d",
                model.global_epoch, best_state["global_epoch"], stale_epochs,
            )
        else:
            current_loss = model.compute_val_prediction_loss(
                val_loader, rng_namespace="warmup_validation", rng_step=0,
            )
            if not math.isfinite(current_loss):
                raise RuntimeError("non-finite initial validation forecast loss")
            current_state = _capture(model, train_loader, history)
            initial_state = current_state
            best_state = current_state
            best_loss = current_loss
            stale_epochs = 0
            ctx.state["warmup_initial_snapshot"] = initial_state
            ctx.log_metrics({"warmup": {
                "val_pred_loss": current_loss,
                "best_val_pred_loss": best_loss,
                "stale_epochs": stale_epochs,
            }}, step=0)
            logger.info("warm-up epoch=0 val_pred_loss=%.9g", current_loss)
            ctx.state["warmup_snapshot"] = current_state
            ctx.state["warmup_pred_loss"] = current_loss
            ctx.state["warmup_best_snapshot"] = best_state
            ctx.state["warmup_best_pred_loss"] = best_loss
            ctx.state["warmup_stale_epochs"] = stale_epochs
            try:
                ctx.checkpoint.save(step=0)
            except RuntimeError:
                pass

        while (
            stale_epochs < patience
            and (max_epochs is None or model.global_epoch < max_epochs)
        ):
            model.fit(
                data_loader=train_loader,
                phase_name="shared_warmup",
                num_epoch=1,
                applied_update_observer=record_update,
            )
            epoch = model.global_epoch
            current_loss = model.compute_val_prediction_loss(
                val_loader, rng_namespace="warmup_validation", rng_step=0,
            )
            if not math.isfinite(current_loss):
                raise RuntimeError(
                    f"non-finite warm-up validation forecast loss at epoch {epoch}"
                )
            current_state = _capture(model, train_loader, history)
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = current_state
                stale_epochs = 0
            else:
                stale_epochs += 1
            ctx.log_metrics({"warmup": {
                "val_pred_loss": current_loss,
                "best_val_pred_loss": best_loss,
                "stale_epochs": stale_epochs,
            }}, step=epoch)
            logger.info(
                "warm-up epoch=%d val_pred_loss=%.9g best_epoch=%d "
                "best_loss=%.9g stale_epochs=%d/%d",
                epoch, current_loss, best_state["global_epoch"],
                best_loss, stale_epochs, patience,
            )
            ctx.state["warmup_snapshot"] = current_state
            ctx.state["warmup_pred_loss"] = current_loss
            ctx.state["warmup_best_snapshot"] = best_state
            ctx.state["warmup_best_pred_loss"] = best_loss
            ctx.state["warmup_stale_epochs"] = stale_epochs
            try:
                ctx.checkpoint.save(step=epoch)
            except RuntimeError:
                pass

        stopped_by_patience = stale_epochs >= patience
        logger.info(
            "warm-up finished: stop_epoch=%d best_epoch=%d "
            "best_loss=%.9g stale_epochs=%d reason=%s",
            current_state["global_epoch"], best_state["global_epoch"],
            best_loss, stale_epochs,
            "patience" if stopped_by_patience else "warmup_max_epochs",
        )
        return {
            "stop_state": current_state,
            "best_state": best_state,
            "initial_state": initial_state,
            "stop_epoch": current_state["global_epoch"],
            "stop_val_pred_loss": current_loss,
            "best_epoch": best_state["global_epoch"],
            "best_val_pred_loss": best_loss,
            "stale_epochs": stale_epochs,
            "stopped_by_patience": stopped_by_patience,
        }


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
            declaration["tip"] = True
        return declaration

    def process(self, warmup_state, ctx):
        from src.build import (
            build_dataloaders,
            configure_deterministic_runtime,
            init_joint_model,
        )
        from src.experiments import _run_b2
        from src.pipeline import prepare_dataset

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
        if cfg.has_section("tip"):
            cfg.set("tip", "warmup_epochs", "0")
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


def build_staged_pipeline() -> Pipeline:
    return Pipeline(
        [PrepareData, Warmup, ContinueTraining, Evaluate],
        name="impute-then-forecast-staged",
    )
