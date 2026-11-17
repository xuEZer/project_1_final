"""Evaluate the final continuation state on every chronological split."""

import logging

import torch
from expman import Stage

from src.artifacts import output_dir, state_hash
from src.build import configure_deterministic_runtime, init_joint_model
from src.measurement import ForecastEngine, write_json
from src.runtime import build_config, prepare_dataset, runtime_from_dataset
from src.training import evaluate_loss, seed_model


class Evaluate(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {
            key: True
            for key in (
                "dataset",
                "data",
                "missing_seed",
                "seed",
                "device",
                "models",
                "paths",
                "training",
            )
        }

    def process(self, trained, ctx):
        summary = trained["summary"]
        if not summary["training_complete"]:
            raise ValueError("Cannot evaluate an unfinished training state")
        dataset = prepare_dataset(ctx.cfg)
        logger = logging.getLogger(__name__)
        configure_deterministic_runtime(ctx.cfg["seed"], logger)
        seed_model(ctx.cfg["seed"])
        model = init_joint_model(
            build_config(ctx.cfg, data=runtime_from_dataset(dataset)),
        )
        model.imputer.load_state_dict(trained["final"])
        evaluated_hash = state_hash(model.imputer.state_dict())
        if evaluated_hash != summary["final_imputer_hash"]:
            raise ValueError("Evaluation weights do not match the final training state")
        model.imputer.eval()
        engine = ForecastEngine(model, dataset, ctx.cfg["training"]["batch_size"])
        scores = {}
        try:
            for split in ("train", "val", "test"):
                scores[split] = {"prediction_mse": evaluate_loss(engine, split)}
                data = dataset[split]
                total = 0.0
                count = 0
                with torch.no_grad():
                    for start in range(0, len(data["X_N"]), engine.batch_size):
                        section = slice(start, start + engine.batch_size)
                        x = torch.as_tensor(data["X_N"][section], device=model.device)
                        mask = torch.as_tensor(
                            data["X_N_mask"][section],
                            device=model.device,
                            dtype=torch.float32,
                        )
                        imputed = engine.complete(x, mask).double()
                        target = torch.as_tensor(
                            data["X"][section], device=model.device, dtype=torch.float64
                        )
                        hidden = torch.as_tensor(
                            data["X_eval_mask"][section], device=model.device
                        )
                        total += float(((imputed - target).square() * hidden).sum())
                        count += int(hidden.sum())
                if count:
                    scores[split]["imputation_mse"] = total / count
            ctx.log_metrics(
                {
                    "evaluation": scores,
                    "selection": {"step": summary["completed_steps"]},
                }
            )
        finally:
            engine.close()
        result = {
            "training": summary,
            "checkpoint_selection": "final",
            "evaluated_imputer_hash": evaluated_hash,
            "evaluated_splits": ["train", "val", "test"],
            "metrics_store": "metrics.sqlite3",
        }
        write_json(output_dir(ctx) / "evaluation.json", result)
        return result
