"""Plot validation forecast MSE from expman metrics into paper/figures."""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def validation_curves(batch, run_ids):
    """Read the latest recorded attempt for each run and validation epoch."""
    database = batch / "metrics.sqlite3"
    with sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT run_id, step, value, attempt FROM metrics "
            "WHERE json_extract(metric_path, '$[0]') = 'validation' "
            "AND json_extract(metric_path, '$[1]') = 'prediction_mse' "
            "AND step >= 0 ORDER BY run_id, step, attempt"
        )
        curves = {}
        for run_id, step, value, _attempt in rows:
            if not run_ids or run_id in run_ids:
                curves.setdefault(run_id, {})[step] = value
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--name", default="validation")
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        parser.error("name must be an alphanumeric filename")
    figure, axes = plt.subplots(figsize=(7, 4), layout="constrained")
    count = 0
    for batch in args.batches:
        for run_id, curve in validation_curves(batch, args.run_id).items():
            epochs = sorted(curve)
            label = f"{batch.name}/{run_id}"
            metadata = batch / "experiments" / run_id / "artifacts" / "metadata.json"
            if metadata.is_file():
                config = json.loads(metadata.read_text())["config"]
                label = f"{config['dataset']['name']} / {config['models']['imputer']['model']} / {config['models']['predictor']['model']} / {config['training']['method']} / {run_id}"
            axes.plot(epochs, [curve[epoch] for epoch in epochs], label=label)
            count += 1
    if count == 0:
        parser.error("No validation metrics match the selected runs")
    axes.set(xlabel="Epoch", ylabel="Validation forecast MSE")
    axes.legend(fontsize=6)
    axes.grid(alpha=0.2)
    output = Path(__file__).resolve().parents[1] / "paper/figures"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / (args.name + ".png"), dpi=180)
    figure.savefig(
        output / (args.name + ".pdf"), metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
