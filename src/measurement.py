"""Forward-query measurements of the empirical forecast objective."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import torch


def flat(values):
    """Concatenate detached parameter-shaped tensors in parameter order."""
    return torch.cat([value.detach().reshape(-1) for value in values])


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def write_json(path, value):
    """Atomically write a configuration or artifact manifest."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dot(a, b):
    return float((a.double() * b.double()).sum())


class ForecastEngine:
    """Query a frozen forecaster with the shared training batch size."""

    def __init__(self, model, dataset, batch_size):
        self.model = model
        self.params = list(model.imputer.parameters())
        self.batch_size = batch_size
        self.splits = {
            name: {
                key: torch.as_tensor(
                    source[key], device=model.device, dtype=torch.float32
                )
                for key in (
                    "X_N",
                    "X_N_mask",
                    "X_NA",
                    "X_NA_mask",
                    "X_A_mask",
                    "Y_N",
                    "Y_N_mask",
                )
            }
            for name, source in dataset.items()
        }
        self.calls = self.windows = 0
        self._hook = model.predictor.register_forward_pre_hook(self._count)

    def close(self):
        self._hook.remove()

    def _count(self, module, args):
        self.calls += 1
        self.windows += len(args[0])

    @torch.no_grad()
    def set_weights(self, weights, displacement=None):
        """Set theta, or theta minus a descent displacement."""
        offset = 0
        for parameter, original in zip(self.params, weights, strict=True):
            parameter.copy_(original)
            if displacement is not None:
                count = parameter.numel()
                parameter.sub_(displacement[offset : offset + count].view_as(parameter))
                offset += count

    @staticmethod
    def losses(prediction, future, missing):
        """Return one mean squared error per window over observed future targets."""
        observed = 1 - missing
        counts = observed.sum((1, 2))
        if torch.any(counts == 0):
            raise ValueError("Every forecast window must have an observed target")
        values = ((prediction.double() - future.double()).square() * observed).sum(
            (1, 2)
        ) / counts
        if not torch.isfinite(values).all():
            raise ArithmeticError("Nonfinite forecast loss")
        return values

    @torch.no_grad()
    def complete(self, inputs, missing):
        """Complete windows with the same imputer query semantics as PILOT."""
        if getattr(self.model.imputer, "windowwise_pilot_queries", False):
            return torch.cat(
                [
                    self.model.imputer(window, mask)["imp_data"]
                    for window, mask in zip(
                        inputs.split(1), missing.split(1), strict=True
                    )
                ]
            )
        return self.model.imputer(inputs, missing)["imp_data"]

    @torch.no_grad()
    def full_loss(self, split="train"):
        data = self.splits[split]
        population = len(data["X_N"])
        if population == 0:
            raise ValueError(f"The {split} split contains no forecast windows")
        total = 0.0
        for start in range(0, population, self.batch_size):
            section = slice(start, start + self.batch_size)
            inputs, missing = data["X_N"][section], data["X_N_mask"][section]
            completed = self.complete(inputs, missing)
            prediction = self.model.predictor(completed)["pred_data"]
            total += float(
                self.losses(
                    prediction, data["Y_N"][section], data["Y_N_mask"][section]
                ).sum()
            )
        if not math.isfinite(total):
            raise ArithmeticError("Nonfinite full-split forecast loss")
        return total / population

    @torch.no_grad()
    def coordinates(self, ids, basis, epsilon, step):
        """Return paired central differences for each queried window and axis."""
        data = self.splits["train"]
        rows = []
        for chunk_index, indices in enumerate(
            ids.to(self.model.device).split(self.batch_size)
        ):
            rows.append(
                self.model.prediction_coordinates(
                    data["X_N"][indices],
                    data["X_N_mask"][indices],
                    data["Y_N"][indices],
                    data["Y_N_mask"][indices],
                    basis,
                    epsilon,
                    step,
                    chunk_index,
                )
            )
        values = torch.cat(rows)
        if not torch.isfinite(values).all():
            raise ArithmeticError("Nonfinite forecast finite difference")
        return values
