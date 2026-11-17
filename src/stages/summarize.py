"""Record the artifacts produced by an experiment."""

from expman import Stage

from src.artifacts import output_dir
from src.measurement import write_json


class Summarize(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, result, ctx):
        write_json(output_dir(ctx) / "summary.json", result)
        return result
