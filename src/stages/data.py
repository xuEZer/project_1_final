"""Immutable, shared chronological data preparation."""

from expman import Stage

from src.runtime import prepare_dataset


class PrepareData(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {
            "dataset": True,
            "data": True,
            "missing_seed": True,
            "paths": True,
        }

    def process(self, data, ctx):
        dataset = prepare_dataset(ctx.cfg)
        ctx.log_metrics(
            {"data": {"windows": {k: len(v["X_N"]) for k, v in dataset.items()}}}
        )
        return dataset
