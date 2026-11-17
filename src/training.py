"""Reconstruction training with PILOT and its component ablations."""

import logging
import math
import random
import time

import numpy as np
import torch

from src.artifacts import (
    archive_sources,
    output_dir,
    provenance,
    save_weights,
    state_hash,
)
from src.measurement import ForecastEngine, flat, restore_rng, rng_state, write_json
from src.models.jointModel import JointModel
from src.rng import derive_seed, torch_seed_scope
from src.runtime import _cpu_copy, _plain
from src.updates import candidate, retain_history

VALIDATION_PATIENCE = 5
CHECKPOINT_INTERVAL = 20
METHODS = {"raw", "pilot", "random", "point", "mean_gate", "forecast_direction"}


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(cfg):
    cfg = _plain(cfg)
    training = cfg["training"]
    if set(training) != {"method", "lr", "epochs", "batch_size"}:
        raise ValueError("training requires only method, lr, epochs, and batch_size")
    if training["method"] not in METHODS:
        raise ValueError("Unknown training method")
    if type(training["epochs"]) is not int or not 1 <= training["epochs"] <= 50:
        raise ValueError("Training epochs must be an integer from 1 to 50")
    if type(training["batch_size"]) is not int or training["batch_size"] < 1:
        raise ValueError("Batch size must be a positive integer")
    if not math.isfinite(training["lr"]) or training["lr"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    pilot = cfg["pilot"]
    if set(pilot) != {"subspace_rank", "probe_batches", "epsilon", "delta"}:
        raise ValueError(
            "pilot requires subspace_rank, probe_batches, epsilon, and delta"
        )
    for key in ("subspace_rank", "probe_batches"):
        if type(pilot[key]) is not int or pilot[key] < 1:
            raise ValueError(f"pilot.{key} must be a positive integer")
    if not math.isfinite(pilot["epsilon"]) or pilot["epsilon"] <= 0:
        raise ValueError("Finite-difference epsilon must be finite and positive")
    if not math.isfinite(pilot["delta"]) or not 0 < pilot["delta"] < 1:
        raise ValueError("Miscoverage delta must be between zero and one")
    if "measurement" in cfg:
        raise ValueError("Forecast measurements are defined by the selected method")
    if cfg["device"] != [0]:
        raise ValueError("Experiments use GPU 0")


def buffers(model):
    return {key: value.detach().clone() for key, value in model.named_buffers()}


def restore_buffers(model, saved):
    with torch.no_grad():
        for key, value in model.named_buffers():
            value.copy_(saved[key])


def evaluate_loss(engine, split):
    """Evaluate the full split without changing training state or its RNG stream."""
    modes = JointModel._capture_training_modes(engine.model.imputer)
    saved = buffers(engine.model.imputer)
    try:
        engine.model.imputer.eval()
        with torch_seed_scope(engine.model.base_seed, "evaluation", split):
            return engine.full_loss(split)
    finally:
        restore_buffers(engine.model.imputer, saved)
        JointModel._restore_training_modes(modes)


@torch.no_grad()
def synchronize_first_moment(optimizer, parameters, weights, applied):
    """Set AdamW's first moment to imply the applied displacement at theta.

    For d = lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta),
    solve for m while preserving the second moment and step counter.
    """
    group = optimizer.param_groups[0]
    learning_rate = group["lr"]
    beta1, beta2 = group["betas"]
    offset = 0
    for parameter, before in zip(parameters, weights, strict=True):
        count = parameter.numel()
        displacement = applied[offset : offset + count].view_as(parameter)
        offset += count
        state = optimizer.state.get(parameter)
        if not state:
            if torch.count_nonzero(displacement):
                raise ArithmeticError("A corrected parameter has no AdamW state")
            continue
        step = int(state["step"].item())
        variance = state["exp_avg_sq"] / (1 - beta2**step)
        target = displacement / learning_rate - group["weight_decay"] * before
        state["exp_avg"].copy_(
            target * (variance.sqrt() + group["eps"]) * (1 - beta1**step)
        )


def run_training(model, dataset, cfg, ctx):
    cfg = _plain(cfg)
    folder = output_dir(ctx)
    training = cfg["training"]
    settings = cfg["pilot"]
    method = training["method"]
    seed = int(cfg["seed"])
    optimizer = torch.optim.AdamW(
        model.imputer.parameters(),
        lr=float(training["lr"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    engine = ForecastEngine(model, dataset, training["batch_size"])
    train = engine.splits["train"]
    population = len(train["X_N"])
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    steps_per_epoch = math.ceil(population / batch_size)
    planned = epochs * steps_per_epoch
    history = []
    global_step = completed_epochs = bad_epochs = 0
    best_loss = float("inf")
    totals = {"interventions": 0, "zero_updates": 0, "rank_bypasses": 0}
    initial_hash = state_hash(model.imputer.state_dict())
    predictor_hash = state_hash(model.predictor.state_dict())
    saved = ctx.checkpoint_state
    if saved is not None:
        model.imputer.load_state_dict(saved["imputer"])
        optimizer.load_state_dict(saved["optimizer"])
        history = [value.to(model.device) for value in saved["history"]]
        global_step = saved["global_step"]
        completed_epochs = saved["completed_epochs"]
        bad_epochs, best_loss = saved["bad_epochs"], saved["best_loss"]
        totals = saved["totals"]
        initial_hash = saved["initial_hash"]
        engine.calls, engine.windows = saved["calls"], saved["windows"]
        restore_rng(saved["rng"])
    else:
        best_loss = evaluate_loss(engine, "val")
        ctx.log_metrics(
            {
                "validation": {"prediction_mse": best_loss},
                "full_training": {"prediction_mse": evaluate_loss(engine, "train")},
            },
            step=0,
        )
    meta = provenance(cfg)
    meta.update(
        train_windows=population,
        planned_steps=planned,
        initial_imputer_hash=initial_hash,
        predictor_hash=predictor_hash,
        run_id=ctx.run_id,
    )
    write_json(folder / "metadata.json", meta)
    archive_sources(folder)

    def persist():
        ctx.checkpoint(
            global_step,
            {
                "imputer": _cpu_copy(model.imputer.state_dict()),
                "optimizer": _cpu_copy(optimizer.state_dict()),
                "history": _cpu_copy(history),
                "global_step": global_step,
                "completed_epochs": completed_epochs,
                "rng": rng_state(),
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "totals": dict(totals),
                "initial_hash": initial_hash,
                "calls": engine.calls,
                "windows": engine.windows,
            },
        )

    try:
        for epoch in range(global_step // steps_per_epoch, epochs):
            if bad_epochs >= VALIDATION_PATIENCE:
                break
            order = torch.randperm(
                population,
                generator=torch.Generator().manual_seed(
                    derive_seed(seed, "train_order", epoch)
                ),
            )
            for batch_index, ids_cpu in enumerate(order.split(batch_size)):
                if epoch * steps_per_epoch + batch_index < global_step:
                    continue
                model._synchronize_for_timing()
                started = time.perf_counter()
                ids = ids_cpu.to(model.device)
                model.imputer.train()
                optimizer.zero_grad(set_to_none=True)
                weights = [parameter.detach().clone() for parameter in engine.params]
                loss = model.imputer(
                    train["X_NA"][ids],
                    train["X_NA_mask"][ids],
                    train["X_N"][ids],
                    train["X_A_mask"][ids],
                )["imp_loss"]
                if not torch.isfinite(loss):
                    raise ArithmeticError("Nonfinite reconstruction loss")
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in engine.params
                ):
                    raise ArithmeticError("Nonfinite reconstruction gradient")
                optimizer.step()
                proposed = [parameter.detach().clone() for parameter in engine.params]
                raw = flat(
                    [
                        before - after
                        for before, after in zip(weights, proposed, strict=True)
                    ]
                )
                optimizer.zero_grad(set_to_none=True)
                state = rng_state()
                saved_buffers = buffers(model.imputer)
                engine.set_weights(weights)
                model.imputer.eval()
                try:
                    selected, info = candidate(
                        engine,
                        weights,
                        raw,
                        history,
                        ids_cpu,
                        settings,
                        method,
                        batch_size,
                        global_step,
                        seed,
                    )
                finally:
                    restore_buffers(model.imputer, saved_buffers)
                    restore_rng(state)
                    model.imputer.train()
                if torch.equal(selected, raw):
                    engine.set_weights(proposed)
                else:
                    engine.set_weights(weights, selected)
                actual = flat(
                    [
                        before - parameter.detach()
                        for before, parameter in zip(
                            weights, engine.params, strict=True
                        )
                    ]
                )
                if not torch.equal(selected, raw):
                    synchronize_first_moment(optimizer, engine.params, weights, actual)
                history = retain_history(history, actual, settings["subspace_rank"] - 1)
                global_step += 1
                totals["interventions"] += int(not info["released"])
                totals["zero_updates"] += int(not bool(torch.count_nonzero(actual)))
                totals["rank_bypasses"] += int(info["reason"] == "insufficient_rank")
                model._synchronize_for_timing()
                values = {
                    key: value
                    for key, value in info.items()
                    if isinstance(value, (int, float))
                }
                values.update(
                    epoch=epoch + 1,
                    batch_windows=len(ids),
                    reconstruction_loss=float(loss.detach()),
                    raw_displacement_norm=float(raw.double().norm()),
                    applied_displacement_norm=float(actual.double().norm()),
                    correction_norm=float((actual - raw).double().norm()),
                    seconds=time.perf_counter() - started,
                    predictor_calls_cumulative=engine.calls,
                    window_forecasts_cumulative=engine.windows,
                )
                ctx.log_metrics({"training": values}, step=global_step)
                epoch_end = batch_index == steps_per_epoch - 1
                if epoch_end:
                    completed_epochs = epoch + 1
                    validation = evaluate_loss(engine, "val")
                    if validation < best_loss:
                        best_loss, bad_epochs = validation, 0
                    else:
                        bad_epochs += 1
                    ctx.log_metrics(
                        {
                            "validation": {"prediction_mse": validation},
                            "full_training": {
                                "prediction_mse": evaluate_loss(engine, "train")
                            },
                            "early_stopping": {"bad_epochs": bad_epochs},
                        },
                        step=completed_epochs,
                    )
                    logging.getLogger(__name__).info(
                        "epoch=%d/%d method=%s validation_mse=%.9g",
                        completed_epochs,
                        epochs,
                        method,
                        validation,
                    )
                if epoch_end or global_step % CHECKPOINT_INTERVAL == 0:
                    persist()
                ctx.report_progress(global_step, total=planned, unit="step")
        stopped_early = global_step < planned and bad_epochs >= VALIDATION_PATIENCE
        if global_step != planned and not stopped_early:
            raise AssertionError("Training ended before satisfying its stopping rule")
        if state_hash(model.predictor.state_dict()) != predictor_hash:
            raise AssertionError("Frozen predictor state changed")
        final = _cpu_copy(model.imputer.state_dict())
        save_weights(folder / "final_state.pt", final)
        summary = {
            "training_complete": True,
            "completed_steps": global_step,
            "planned_steps": planned,
            "completed_epochs": completed_epochs,
            "method": method,
            "initial_imputer_hash": initial_hash,
            "final_imputer_hash": state_hash(final),
            "predictor_unchanged": True,
            "checkpoint_selection": "final",
            "stopped_early": stopped_early,
            "stop_reason": "early_stopping" if stopped_early else "max_epochs",
            "early_stopping_patience": VALIDATION_PATIENCE,
        }
        ctx.log_metrics(
            {
                "summary": {
                    **totals,
                    "completed_steps": global_step,
                    "completed_epochs": completed_epochs,
                    "predictor_calls": engine.calls,
                    "window_forecasts": engine.windows,
                }
            }
        )
        write_json(folder / "training_summary.json", summary)
        return {
            "final": final,
            "summary": summary,
            "artifact_dir": str(folder),
            "source_digest": meta["source_digest"],
        }
    finally:
        engine.close()
