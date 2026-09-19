#!/usr/bin/env python3
"""Pilot: does TIP's per-step protection transfer to the endpoint at MATCHED steps?

The existing experiment compares B1 and B2 at their own early-stopping points,
which differ in step count in every pair, so a better endpoint cannot be told
apart from "trained longer".  This pilot removes that confound:

  * both arms run the full epoch budget (the callback never early-stops),
  * val AND test forecast MSE are recorded at every epoch, hence at matched
    optimizer steps, for both arms.

Output: one CSV row per (arm, epoch), plus a matched-step comparison table.
With a matched table the failure mode becomes readable:

  * B2 below B1 in test MSE at the same step  -> the drift transfers, the
    problem is in the dynamics (correction never accumulates),
  * B2 only leads at unmatched points         -> it is a training-length effect
    and adding a task anchor would not help.

READ-ONLY with respect to the repository: it imports the production training
entry points (`_run_b1` / `_run_b2`) unchanged and injects evaluation through
their `on_epoch_end` hook.  Nothing under `src/` or `configs/` is modified.

Usage:
  python scripts/pilot_matched_trajectory.py --dataset azure2019_I_5T \
      --imputer SAITS --predictor Toto2 --epochs 20 --eval-every 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from expman import load_configs  # noqa: E402

from src.build import (  # noqa: E402
    build_dataloaders, configure_deterministic_runtime, init_joint_model,
    parse_column_index,
)
from src.data_loader import load_h5, preprocess_dataset  # noqa: E402
from src.evaluator import evaluate  # noqa: E402
from src.experiments import _run_b1, _run_b2  # noqa: E402
from src.logger import setup_logger  # noqa: E402
from src.pipeline import (  # noqa: E402
    build_config, run_tag, runtime_from_dataset,
)


class LimitedLoader:
    """Yield at most ``limit`` batches of a DataLoader.

    Caps the cost of a mid-training evaluation without touching the evaluator.
    Evaluation loaders are not shuffled, so the same subset is used every time
    and stays comparable across arms.
    """

    def __init__(self, loader, limit: int | None) -> None:
        self._loader = loader
        self._limit = limit

    def __iter__(self):
        for index, batch in enumerate(self._loader):
            if self._limit is not None and index >= self._limit:
                break
            yield batch

    def __len__(self) -> int:
        base = len(self._loader)
        return base if self._limit is None else min(base, self._limit)


def build(args: argparse.Namespace, logger: logging.Logger, tb):
    """Fresh model + dataloaders from one run of the experiment YAML.

    Both arms are rebuilt from the same run dictionary, so they share the
    initialization seed, the data order and the whole protocol.
    """
    run = load_configs(args.config)[args.index]
    run = json.loads(json.dumps(run))          # unfreeze into a plain dict
    run["training"]["max_epochs"] = args.epochs
    # patience <= 0 is the fixed-horizon mode: neither the built-in monitor nor
    # our own callback may early-stop, so both arms stay step-aligned.
    run["training"]["patience"] = 0
    if args.anchor_delta:
        run["tip"]["anchor_delta"] = args.anchor_delta

    ds = run["dataset"]
    folder = preprocess_dataset(
        source=ds["source"], repo_name=ds["repo_name"],
        dataset_name=ds["dataset_name"], frequency=ds["frequency"],
        item_id=ds["item_id"], column_index=parse_column_index(str(ds["column_index"])),
        missing_rate=run["data"]["missing_rate"],
        artificial_missing_rate=run["data"]["artificial_missing_rate"],
        block_size=run["data"]["block_size"], lookback_len=run["data"]["lookback"],
        pred_len=run["data"]["pred_len"], val_ratio=run["data"]["val_ratio"],
        test_ratio=run["data"]["test_ratio"], random_seed=run.get("missing_seed", run["seed"]),
        logger=logger, save_path=Path(run["paths"]["cache_dir"]) / "data",
    )
    dataset = load_h5(folder / "dataset.h5")
    cfg = build_config(run, data=runtime_from_dataset(dataset))
    model = init_joint_model(cfg, logger, tb)
    train_loader, test_loader, no_gt = build_dataloaders(
        config=cfg, seed=run["seed"], dataset=dataset,
        joint_model=model, logger=logger,
    )
    return cfg, model, train_loader, test_loader, no_gt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"),
                    help="experiment YAML the run is taken from")
    ap.add_argument("--index", type=int, default=0,
                    help="which expanded run of that YAML to use")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=2,
                    help="evaluate val+test every K epochs")
    ap.add_argument("--eval-batches", type=int, default=0,
                    help="cap evaluation batches per split (0 = all)")
    ap.add_argument("--arms", nargs="+", default=["B1", "B2"])
    ap.add_argument("--anchor-delta", type=float, default=0.0,
                    help="task-anchor magnitude in displacement units (0 = off)")
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "pilot_matched")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    run0 = load_configs(args.config)[args.index]
    tag = f"{run_tag(run0)}_ep{args.epochs}_a{args.anchor_delta:g}"
    logger = setup_logger(args.out / f"{tag}.log", log_name=f"pilot_{tag}")
    configure_deterministic_runtime(args.seed, logger)
    limit = args.eval_batches or None

    rows: list[dict] = []
    for arm in args.arms:
        logger.info("=" * 70)
        logger.info("ARM %s  (%s)", arm, tag)
        logger.info("=" * 70)
        tb = SummaryWriter(log_dir=args.out / f"tb_{arm}")
        cfg, model, train_loader, test_loader, no_gt = build(args, logger, tb)
        val_loader = train_loader.get("val")
        train_eval_loader = train_loader.get("train_eval")

        def on_epoch_end(epoch, train_loss, m, _arm=arm, _tb=tb):
            rec = {"arm": _arm, "epoch": epoch, "step": m.global_optim_step,
                   "train_loss": train_loss}
            if epoch % args.eval_every == 0:
                if train_eval_loader is not None:
                    tr = evaluate(cfg, m, LimitedLoader(train_eval_loader, limit),
                                  no_gt=False, logger=logger, split_name="train")
                    rec["train_imp_mse"] = tr.get("imp_mse")
                    rec["train_pred_mse"] = tr.get("pred_mse")
            if val_loader is not None and epoch % args.eval_every == 0:
                v = evaluate(cfg, m, LimitedLoader(val_loader, limit),
                             no_gt=False, logger=logger, split_name="val")
                rec["val_imp_mse"] = v.get("imp_mse")
                rec["val_pred_mse"] = v.get("pred_mse")
                t = evaluate(cfg, m, LimitedLoader(test_loader, limit),
                             no_gt=no_gt, logger=logger, split_name="test")
                rec["test_imp_mse"] = t.get("imp_mse")
                rec["test_pred_mse"] = t.get("pred_mse")
                for k in ("val_pred_mse", "test_pred_mse"):
                    if rec.get(k) is not None:
                        _tb.add_scalar(f"{k}", rec[k], epoch)
                logger.info("[%s] epoch=%d step=%d val_pred=%s test_pred=%s",
                            _arm, epoch, m.global_optim_step,
                            rec.get("val_pred_mse"), rec.get("test_pred_mse"))
            rows.append(rec)
            return False  # fixed horizon: never early-stop

        if arm == "B1":
            _run_b1(model, train_loader, cfg, logger, on_epoch_end=on_epoch_end)
        else:
            _run_b2(model, train_loader, cfg, logger, on_epoch_end=on_epoch_end,
                    exp_type=arm)
        tb.close()

    csv_path = args.out / f"{tag}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "epoch", "step", "train_loss",
                                           "train_imp_mse", "train_pred_mse",
                                           "val_imp_mse", "val_pred_mse",
                                           "test_imp_mse", "test_pred_mse"])
        w.writeheader()
        w.writerows(rows)

    # ---- matched-step comparison ----------------------------------------
    print(f"\nwrote {csv_path}")
    print(f"\n{'epoch':>6} {'step':>6} | {'B1 tr_pred':>11} {'B2 tr_pred':>11} {'Δtr':>8}"
          f" | {'B1 te_pred':>11} {'B2 te_pred':>11} {'Δte':>8}"
          f" | {'B1 va_pred':>11} {'B2 va_pred':>11} {'Δva':>8}")
    print("-" * 118)
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["epoch"]), r)
    for (arm, epoch), r in sorted(by.items()):
        if arm != "B1":
            continue
        o = by.get(("B2", epoch))
        if not o:
            continue
        def pair(key):
            a, b = r.get(key), o.get(key)
            if a is None or b is None:
                return f"{'--':>11} {'--':>11} {'--':>8}"
            return f"{a:>11.4f} {b:>11.4f} {100*(b/a-1):>7.2f}%"
        print(f"{epoch:>6} {r['step']:>6} | {pair('train_pred_mse')}"
              f" | {pair('test_pred_mse')} | {pair('val_pred_mse')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
