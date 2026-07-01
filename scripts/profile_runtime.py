#!/usr/bin/env python3
"""Profile the main runtime stages of one experiment configuration.

The script intentionally runs only a few batches.  It answers where time is
spent: preprocessing/cache load, model initialization, dataloader transfer,
imputer training, predictor inference, and TIP zero-order estimation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import time
from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import get_tensor_data_loader, load_h5, preprocess_dataset
from src.auto_batch_size import find_max_batch_size
from src.models.imputators import imputator
from src.models.jointModel import JointModel
from src.models.predictors import predictor

GLOBAL_CONFIG_PATH = ROOT / "configs" / "global.ini"


class NullSummaryWriter:
    """Small SummaryWriter stand-in for profiling."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass
class TimingRow:
    stage: str
    seconds: float
    iterations: int = 1
    per_iter_seconds: float | None = None
    notes: str = ""


class RuntimeProfiler:
    def __init__(self, device: str) -> None:
        self.device = device
        self.rows: list[TimingRow] = []

    def sync(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def stage(self, name: str, iterations: int = 1, notes: str = ""):
        self.sync()
        start = time.perf_counter()
        yield
        self.sync()
        elapsed = time.perf_counter() - start
        per_iter = elapsed / iterations if iterations > 0 else None
        self.rows.append(TimingRow(name, elapsed, iterations, per_iter, notes))


def parse_section_overrides(items: list[str]) -> list[tuple[str, str, str]]:
    if len(items) % 2 != 0:
        raise ValueError("Config overrides must be passed as '--section.key value' pairs.")

    overrides: list[tuple[str, str, str]] = []
    for idx in range(0, len(items), 2):
        raw_key = items[idx]
        value = items[idx + 1]
        if not raw_key.startswith("--") or "." not in raw_key[2:]:
            raise ValueError(
                f"Unexpected override '{raw_key}'. Expected '--section.key value'."
            )
        section, key = raw_key[2:].split(".", 1)
        overrides.append((section, key, value))
    return overrides


def apply_overrides(
    config: ConfigParser,
    overrides: list[tuple[str, str, str]],
    *,
    require_existing: bool,
) -> None:
    for section, key, value in overrides:
        if not config.has_section(section):
            if require_existing:
                raise ValueError(f"Section '{section}' not found in config.")
            continue
        if not config.has_option(section, key):
            if require_existing:
                raise ValueError(f"Key '{key}' not found in section '{section}'.")
            continue
        config.set(section, key, value)


def load_profile_config(overrides: list[tuple[str, str, str]]) -> ConfigParser:
    config = ConfigParser(interpolation=ExtendedInterpolation())
    if not GLOBAL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Global config not found: {GLOBAL_CONFIG_PATH}")
    config.read(GLOBAL_CONFIG_PATH)

    # Apply selectors before reading templates.
    selector_overrides = [
        item
        for item in overrides
        if item[:2]
        in {
            ("dataset", "dataset"),
            ("imputator", "model_type"),
            ("predictor", "model_type"),
        }
    ]
    apply_overrides(config, selector_overrides, require_existing=True)

    for section in ["imputator", "predictor"]:
        model_type = config.get(section, "model_type")
        template_path = ROOT / "configs" / "template" / section / f"{model_type}.ini"
        if not template_path.exists():
            raise FileNotFoundError(f"Missing template: {template_path}")
        config.read(template_path)

    dataset_key = config.get("dataset", "dataset")
    template_path = ROOT / "configs" / "template" / "dataset" / f"{dataset_key}.ini"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing dataset template: {template_path}")
    config.read(template_path)

    # Apply all overrides after templates so runtime overrides win.
    apply_overrides(config, overrides, require_existing=True)
    return config


def set_random_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def first_batches(loader, count: int) -> list[tuple[torch.Tensor, ...]]:
    batches = []
    for batch in loader:
        batches.append(tuple(batch))
        if len(batches) >= count:
            break
    return batches


def summarize_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def print_table(rows: list[TimingRow]) -> None:
    print("")
    print("Runtime profile")
    print("-" * 96)
    print(f"{'stage':<34} {'total':>12} {'iters':>8} {'per iter':>12}  notes")
    print("-" * 96)
    for row in rows:
        per_iter = (
            summarize_seconds(row.per_iter_seconds)
            if row.per_iter_seconds is not None
            else "-"
        )
        print(
            f"{row.stage:<34} {summarize_seconds(row.seconds):>12} "
            f"{row.iterations:>8} {per_iter:>12}  {row.notes}"
        )
    print("-" * 96)


def row_avg(rows: list[TimingRow], stage: str) -> float | None:
    for row in rows:
        if row.stage == stage:
            return row.per_iter_seconds
    return None


def print_epoch_estimates(
    rows: list[TimingRow],
    train_batches: int,
    accum_steps: int,
) -> None:
    train = row_avg(rows, "imputer_train_step")
    gp_hat = row_avg(rows, "tip_gp_hat")

    if train is None:
        return

    b1_epoch = train * train_batches
    print("")
    print("Estimated epoch cost")
    print("-" * 96)
    print(f"B1-like epoch: {summarize_seconds(b1_epoch)}")
    if gp_hat is not None:
        optim_steps = math.ceil(train_batches / max(1, accum_steps))
        b2_epoch = train * train_batches + gp_hat * optim_steps
        print(
            f"TIP-like epoch: {summarize_seconds(b2_epoch)} "
            f"(TIP estimator on ~{optim_steps} optimizer steps)"
        )
    print("-" * 96)


def print_top_bottlenecks(rows: list[TimingRow], count: int = 5) -> None:
    measured = sorted(rows, key=lambda row: row.seconds, reverse=True)[:count]
    if not measured:
        return
    print("")
    print("Top bottlenecks")
    print("-" * 96)
    for idx, row in enumerate(measured, start=1):
        print(f"{idx}. {row.stage}: {summarize_seconds(row.seconds)}")
    print("-" * 96)


def profile(args: argparse.Namespace, overrides: list[tuple[str, str, str]]) -> list[TimingRow]:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("profile_runtime")

    profiler = RuntimeProfiler(device="cpu")
    with profiler.stage("config_load"):
        config = load_profile_config(overrides)

    device = config.get("training", "device")
    profiler.device = device
    set_random_seed(config.getint("experiment", "random_seed"))

    print("Profiling configuration")
    print("-" * 96)
    print(f"dataset   : {config.get('dataset', 'dataset')}")
    print(f"imputator : {config.get('imputator', 'model_type')}")
    print(f"predictor : {config.get('predictor', 'model_type')}")
    print(f"exp type  : {config.get('experiment', 'exp_type')}")
    print(f"device    : {device}")
    print(f"batch size: {config.get('training', 'batch_size')}")
    print(f"lookback  : {config.get('dataset', 'lookback_len')}")
    print("-" * 96)

    with profiler.stage("preprocess_dataset", notes="returns quickly on cache hit"):
        dataset_folder = preprocess_dataset(
            dataset_name=config.get("dataset", "dataset_name"),
            frequency=config.get("dataset", "frequency"),
            item_id=config.get("dataset", "item_id"),
            column_index=config.getint("dataset", "column_index"),
            missing_rate=config.getfloat("dataset", "missing_rate"),
            artificial_missing_rate=config.getfloat(
                "dataset", "artificial_missing_rate"
            ),
            block_size=config.getint("dataset", "block_size"),
            lookback_len=config.getint("dataset", "lookback_len"),
            pred_len=config.getint("dataset", "pred_len"),
            val_ratio=config.getfloat("dataset", "val_ratio"),
            test_ratio=config.getfloat("dataset", "test_ratio"),
            random_seed=config.getint("experiment", "random_seed"),
            logger=logger,
            save_path=project_path(config.get("paths", "preprocessed_data_dir")),
        )

    with profiler.stage("load_h5"):
        dataset = load_h5(dataset_folder / "dataset.h5")

    config.set("dataset", "d_feature", str(dataset["train"]["X_N"].shape[2]))
    config.set("dataset", "lookback_len", str(dataset["train"]["X_N"].shape[1]))

    batch_size = config.getint("training", "batch_size")
    dataloader_num_workers = config.getint(
        "training",
        "dataloader_num_workers",
        fallback=0,
    )
    pin_memory = (
        config.getboolean(
            "training",
            "pin_memory",
            fallback=str(device).startswith("cuda"),
        )
        and str(device).startswith("cuda")
    )

    with profiler.stage(
        "build_dataloaders",
        notes=f"workers={dataloader_num_workers}, pin_memory={pin_memory}",
    ):
        train_loader, eval_loader, no_gt = get_tensor_data_loader(
            dataset,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
        )

    with contextlib.ExitStack():
        with profiler.stage("model_init", notes="includes frozen TSFM load/compile"):
            joint_model = JointModel(
                imputer=imputator(config.get("imputator", "model_type"), config),
                predictor=predictor(config.get("predictor", "model_type"), config),
                optimizer_name=config.get("training", "optimizer"),
                lr=config.getfloat("training", "lr"),
                device=device,
                tb_summary_writer=NullSummaryWriter(),
                logger=logger,
            )
            joint_model.to(device)

        accum_steps = 1
        manual_micro_batch = config.getint("training", "micro_batch_size", fallback=0)
        if manual_micro_batch > 0:
            accum_steps = math.ceil(batch_size / manual_micro_batch)
        joint_model.accum_steps = accum_steps

        train_batches = first_batches(
            train_loader["train"],
            max(args.batches + args.warmup_batches, 1),
        )
        sample_batches = train_batches[args.warmup_batches :]

        if not sample_batches:
            raise RuntimeError("No training batches available for profiling.")

        if args.include_auto_batch:
            with profiler.stage(
                "auto_batch_size_search",
                notes="loads/measures candidates in subprocesses",
            ):
                best_batch_size = find_max_batch_size(
                    joint_model,
                    train_batches[0],
                    batch_size,
                    device,
                    logger,
                    config,
                )
            profiler.rows[-1].notes += f", best={best_batch_size}"

        if args.warmup_batches > 0:
            with profiler.stage(
                "warmup",
                iterations=min(args.warmup_batches, len(train_batches)),
                notes="not included in per-stage averages",
            ):
                joint_model.imputer.train()
                for batch in train_batches[: args.warmup_batches]:
                    X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
                        joint_model._move_batch_to_device(batch)
                    )
                    output = joint_model.forward(
                        X_NA,
                        X_NA_mask,
                        X_N,
                        X_A_mask,
                        Y_N,
                        Y_N_mask,
                        run_predictor=False,
                    )
                    loss = output["loss"]
                    loss.backward()
                    joint_model.optimizer.zero_grad(set_to_none=True)

        with profiler.stage(
            "host_to_device",
            iterations=len(sample_batches),
            notes="batch tensor transfer only",
        ):
            moved_batches = [
                joint_model._move_batch_to_device(batch) for batch in sample_batches
            ]

        with profiler.stage("imputer_forward", iterations=len(moved_batches)):
            joint_model.imputer.eval()
            with torch.no_grad():
                for X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask in moved_batches:
                    joint_model.forward(
                        X_NA,
                        X_NA_mask,
                        X_N,
                        X_A_mask,
                        Y_N,
                        Y_N_mask,
                        run_predictor=False,
                    )

        with profiler.stage("imputer_train_step", iterations=len(moved_batches)):
            joint_model.imputer.train()
            for X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask in moved_batches:
                joint_model.optimizer.zero_grad(set_to_none=True)
                output = joint_model.forward(
                    X_NA,
                    X_NA_mask,
                    X_N,
                    X_A_mask,
                    Y_N,
                    Y_N_mask,
                    run_predictor=False,
                )
                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_(joint_model.imputer.parameters(), 10.0)
                joint_model.optimizer.step()

        first = moved_batches[0]
        X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = first
        with torch.no_grad():
            imputed_X = joint_model.imputer(X_NA, X_NA_mask, X_N, X_A_mask)[
                "imp_data"
            ]

        if not args.skip_predictor:
            with profiler.stage(
                "predictor_forward",
                iterations=args.predictor_repeats,
                notes="frozen TSFM only",
            ):
                for _ in range(args.predictor_repeats):
                    joint_model.predictor(imputed_X)

            with profiler.stage("compute_Q", iterations=args.q_repeats):
                for _ in range(args.q_repeats):
                    joint_model._compute_Q(
                        X_NA,
                        X_NA_mask,
                        X_N,
                        X_A_mask,
                        Y_N,
                        Y_N_mask,
                    )

        if not args.skip_tip:
            tip_k = args.tip_k or config.getint("training", "tip_k", fallback=20)
            with profiler.stage(
                "tip_gp_hat",
                iterations=args.tip_repeats,
                notes=f"k={tip_k}, calls predictor 2*k times per iter",
            ):
                for _ in range(args.tip_repeats):
                    joint_model._compute_gp_hat(
                        X_NA,
                        X_NA_mask,
                        X_N,
                        X_A_mask,
                        Y_N,
                        Y_N_mask,
                        k=tip_k,
                        epsilon=args.tip_epsilon
                        or config.getfloat("training", "tip_epsilon", fallback=1e-4),
                    )

        print_table(profiler.rows)
        print_top_bottlenecks(profiler.rows)
        print_epoch_estimates(
            profiler.rows,
            train_batches=len(train_loader["train"]),
            accum_steps=accum_steps,
        )
        print("")
        print(f"train batches: {len(train_loader['train'])}")
        print(f"test batches : {len(eval_loader)}")
        print(f"test no_gt   : {no_gt}")

        if args.output_json:
            payload = [asdict(row) for row in profiler.rows]
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote JSON profile: {args.output_json}")

    return profiler.rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile major runtime stages for one experiment config. "
            "Pass normal config overrides after the profiling options, e.g. "
            "--dataset.dataset azure2019_D_5T --predictor.model_type TimesFM25."
        )
    )
    parser.add_argument("--batches", type=int, default=3, help="Measured train batches.")
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Warmup train batches before measurement.",
    )
    parser.add_argument(
        "--tip-k",
        type=int,
        default=None,
        help="Override k only for the TIP profiling stage.",
    )
    parser.add_argument(
        "--tip-epsilon",
        type=float,
        default=None,
        help="Override epsilon only for the TIP profiling stage.",
    )
    parser.add_argument("--tip-repeats", type=int, default=1)
    parser.add_argument("--q-repeats", type=int, default=1)
    parser.add_argument("--predictor-repeats", type=int, default=1)
    parser.add_argument("--skip-tip", action="store_true")
    parser.add_argument("--skip-predictor", action="store_true")
    parser.add_argument(
        "--include-auto-batch",
        action="store_true",
        help="Also time the auto batch-size search. This can be expensive.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    overrides = parse_section_overrides(unknown)
    profile(args, overrides)


if __name__ == "__main__":
    main()
