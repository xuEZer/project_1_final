"""Train an imputer with a frozen forecasting model."""

import logging

from expman import Stage

from src.build import configure_deterministic_runtime, init_joint_model
from src.runtime import build_config, runtime_from_dataset
from src.training import run_training, seed_model, validate_config


class Train(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {
            k: True
            for k in (
                "dataset",
                "data",
                "missing_seed",
                "seed",
                "device",
                "models",
                "paths",
                "training",
                "pilot",
            )
        }

    def process(self, dataset, ctx):
        validate_config(ctx.cfg)
        logger = logging.getLogger(__name__)
        configure_deterministic_runtime(ctx.cfg["seed"], logger)
        seed_model(ctx.cfg["seed"])
        cfg = build_config(ctx.cfg, data=runtime_from_dataset(dataset))
        model = init_joint_model(cfg)
        return run_training(model, dataset, ctx.cfg, ctx)
