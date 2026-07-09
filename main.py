"""Main entry point for the FOTSI training and evaluation workflow.

This script orchestrates the full pipeline:
1. Load and merge configuration files.
2. Preprocess and window time-series data.
3. Build dataloaders and initialize model components.
4. Run the configured experiment phase.
5. Evaluate and log final metrics.
"""

import argparse
import logging
import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import math

from src.auto_batch_size import find_max_batch_size
from src.data_loader import (
    get_tensor_data_loader,
    load_h5,
    preprocess_dataset,
)
from src.evaluator import evaluate
from src.experiments import run_experiment
from src.logger import log_config, log_system_info, setup_logger
from src.models.imputators import imputator
from src.models.jointModel import JointModel
from src.models.predictors import predictor
from src.rng import (
    configure_deterministic_backend,
    torch_generator,
    torch_seed_scope,
)

GLOBAL_CONFIG_PATH = Path("./configs/global.ini")


def resolve_batch_size(
    *,
    config: ConfigParser,
    seed: int,
    effective_batch_size: int,
    dataset: dict,
    joint_model: JointModel,
    loader_kwargs: dict,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Determine micro batch size and gradient accumulation steps.

    Three strategies (in priority order):

    1. **Manual** — ``micro_batch_size`` config is non-zero; explicit override.
    2. **Auto** — ternary search over power-of-2 batch sizes for maximum
       throughput via :func:`src.auto_batch_size.find_max_batch_size`.
    3. **Fallback** — ``micro_batch_size = effective_batch_size``, no
       accumulation.

    Returns:
        ``(micro_batch_size, accum_steps)``.

    """
    manual_micro_batch = config.getint(
        "loader", "micro_batch", fallback=0
    )
    if manual_micro_batch > 0:
        micro_batch_size = manual_micro_batch
        accum_steps = math.ceil(effective_batch_size / micro_batch_size)
        logger.info(
            "Using manual micro_batch_size=%d "
            "(target=%d, accum_steps=%d)",
            micro_batch_size,
            effective_batch_size,
            accum_steps,
        )
        return micro_batch_size, accum_steps

    auto_batch = config.has_option("loader", "auto_batch") and config.getboolean(
        "loader", "auto_batch"
    )
    if auto_batch and effective_batch_size > 1:
        auto_probe_generator = torch_generator(
            seed, "auto_batch_probe_loader_shuffle"
        )
        logger.info(
            "Independent RNG: auto_batch_probe_loader_shuffle_seed=%d",
            auto_probe_generator.initial_seed(),
        )
        sample_loader, _, _ = get_tensor_data_loader(
            dataset,
            batch_size=effective_batch_size,
            train_generator=auto_probe_generator,
            **loader_kwargs,
        )
        sample_batch = next(iter(sample_loader["train"]))

        micro_batch_size = find_max_batch_size(
            joint_model,
            sample_batch,
            effective_batch_size,
            config.get("training", "device"),
            logger,
            config,
        )
        accum_steps = math.ceil(effective_batch_size / micro_batch_size)
        logger.info(
            "Effective batch_size: %d | Micro batch_size: %d | "
            "Gradient accumulation steps: %d",
            effective_batch_size,
            micro_batch_size,
            accum_steps,
        )
        return micro_batch_size, accum_steps

    return effective_batch_size, 1


def build_dataloaders(
    config: ConfigParser,
    seed: int,
    dataset: dict,
    joint_model: JointModel,
    logger: logging.Logger,
) -> tuple[dict[str, DataLoader], DataLoader, bool]:
    """Build training and evaluation dataloaders with resolved batch size.

    Reads dataloader config, determines the micro batch size via
    :func:`resolve_batch_size`, sets ``joint_model.accum_steps`` /
    ``joint_model.micro_batch_size``, and constructs train / train-eval /
    test :class:`~torch.utils.data.DataLoader` instances.

    Returns:
        ``(train_loader, eval_loader, no_gt)``.

    """
    effective_batch_size = config.getint("loader", "batch_size")
    device = config.get("training", "device")
    num_workers = config.getint("loader", "num_workers", fallback=0)
    pin_memory = (
        config.getboolean("loader", "pin_memory",
            fallback=str(device).startswith("cuda"),
        )
        and str(device).startswith("cuda")
    )
    loader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}

    micro_batch_size, accum_steps = resolve_batch_size(
        config=config,
        seed=seed,
        effective_batch_size=effective_batch_size,
        dataset=dataset,
        joint_model=joint_model,
        loader_kwargs=loader_kwargs,
        logger=logger,
    )
    joint_model.accum_steps = accum_steps
    joint_model.micro_batch_size = micro_batch_size

    train_shuffle_generator = torch_generator(seed, "train_loader_shuffle")
    logger.info(
        "Independent RNG: train_loader_shuffle_seed=%d",
        train_shuffle_generator.initial_seed(),
    )
    train_loader, eval_loader, no_gt = get_tensor_data_loader(
        dataset,
        batch_size=micro_batch_size,
        train_generator=train_shuffle_generator,
        **loader_kwargs,
    )
    return train_loader, eval_loader, no_gt


def main() -> None:
    """Run the end-to-end experiment lifecycle.

    Args:
        None.

    Returns:
        None.

    """
    config, logger, tb_summary_writer = init_exp()

    log_system_info(logger)
    log_config(config, logger)

    seed = int(config.get("experiment", "seed"))
    configure_deterministic_runtime(seed, logger)

    dataset_folder = preprocess_dataset(
        dataset_name=config.get("dataset", "dataset_name"),
        frequency=config.get("dataset", "frequency"),
        item_id=config.get("dataset", "item_id"),
        column_index=config.getint("dataset", "column_index"),
        missing_rate=config.getfloat("data", "missing_rate"),
        artificial_missing_rate=config.getfloat("data", "artificial_missing_rate"),
        block_size=config.getint("data", "block_size"),
        lookback_len=config.getint("data", "lookback"),
        pred_len=config.getint("data", "pred_len"),
        val_ratio=config.getfloat("data", "val_ratio"),
        test_ratio=config.getfloat("data", "test_ratio"),
        random_seed=config.getint("experiment", "seed"),
        logger=logger,
        save_path=Path(config.get("paths", "cache_dir")) / "data",
    )

    dataset = load_h5(dataset_folder / "dataset.h5")

    config.set("data", "d_feature", str(dataset["train"]["X_N"].shape[2]))
    config.set("data", "lookback", str(dataset["train"]["X_N"].shape[1]))

    joint_model = init_joint_model(config, logger, tb_summary_writer)

    train_loader, eval_loader, no_gt = build_dataloaders(
        config=config,
        seed=seed,
        dataset=dataset,
        joint_model=joint_model,
        logger=logger,
    )

    joint_model = run_experiment(
        exp_type=config["experiment"]["type"],
        joint_model=joint_model,
        train_loader=train_loader,
        config=config,
        logger=logger,
    )
    logger.info("Training completed.")

    for split_name, loader in [
        ("val", train_loader.get("val")),
        ("test", eval_loader),
    ]:
        if loader is None:
            continue
        split_no_gt = getattr(loader, "no_gt", False)
        metrics = evaluate(
            config, joint_model, loader, split_no_gt, logger,
            split_name=split_name,
        )

    logger.info("Experiment completed successfully!")


def init_exp() -> tuple[ConfigParser, logging.Logger, SummaryWriter]:
    """Initialize configuration, logger, and TensorBoard writer.

    Args:
        None.

    Returns:
        A tuple of parsed config, initialized logger, and TensorBoard writer.

    """
    init_logs = []

    config, init_logs = parse_config(init_logs)

    exp_name = config.get("experiment", "name")
    log_save_dir = Path(config.get("paths", "log_dir"))
    exit_if_experiment_completed(exp_name=exp_name, log_save_dir=log_save_dir)

    log_save_dir.mkdir(parents=True, exist_ok=True)
    log_save_path = log_save_dir / f"{exp_name}.log"
    logger = setup_logger(log_save_path, log_name=exp_name)
    logger.info("=" * 70)
    logger.info(f"Exp name: {exp_name}")
    logger.info("=" * 70)
    for msg in init_logs:
        level, text = msg
        if level == "info":
            logger.info(text)
        elif level == "warning":
            logger.warning(text)
        elif level == "error":
            logger.error(text)

    tb_summary_writer = SummaryWriter(log_dir=log_save_dir / exp_name)

    return config, logger, tb_summary_writer


def parse_config(
    init_logs: list[tuple[str, str]],
) -> tuple[ConfigParser, list[tuple[str, str]]]:
    """Load global and template configs, then apply CLI overrides.

    Args:
        init_logs: Mutable list that collects startup log messages.

    Returns:
        Parsed configuration and the updated startup logs.

    """
    config = ConfigParser(interpolation=ExtendedInterpolation())

    if not GLOBAL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Global config file not found: {GLOBAL_CONFIG_PATH}")
    config.read(GLOBAL_CONFIG_PATH)

    argparser = argparse.ArgumentParser()
    _, unknown_args = argparser.parse_known_args()
    for i in range(len(unknown_args)):
        if i % 2 == 0:
            if not unknown_args[i].startswith("--"):
                raise ValueError(
                    f"Unexpected argument format: {unknown_args[i]}. "
                    "Expected '--section.key value'."
                )
            else:
                section_key = unknown_args[i][2:]
                if "." not in section_key:
                    raise ValueError(
                        f"Argument {unknown_args[i]} does not follow 'section.key' format."
                    )
                section, key = section_key.split(".", 1)
                if not config.has_section(section):
                    raise ValueError(f"Section '{section}' not found")
                if not config.has_option(section, key):
                    raise ValueError(f"Key '{key}' not found in section '{section}'")
                if i + 1 >= len(unknown_args) or unknown_args[i + 1].startswith("--"):
                    raise ValueError(f"Missing value for argument {unknown_args[i]}")
                value = unknown_args[i + 1]
                config.set(section, key, value)
                init_logs.append(
                    ("info", f"Overriding config: [{section}] {key} = {value}")
                )

    for section in ["imputer", "predictor"]:
        model = config.get(section, "model")
        template_path = Path(f"./configs/template/{section}/{model}.ini")
        if template_path.exists():
            init_logs.append(
                ("info", f"Loading template config for {model} from {template_path}")
            )
            config.read(template_path)
        else:
            raise FileNotFoundError(
                f"Template config for '{model}' not found at {template_path}"
            )

    dataset_key = config.get("data", "dataset")
    template_path = Path(f"./configs/template/dataset/{dataset_key}.ini")
    if template_path.exists():
        init_logs.append(
            (
                "info",
                f"Loading template config for dataset '{dataset_key}' from {template_path}",
            )
        )
        config.read(template_path)
    else:
        raise FileNotFoundError(
            f"Template config for dataset '{dataset_key}' not found at {template_path}"
        )

    return config, init_logs


def exit_if_experiment_completed(exp_name: str, log_save_dir: Path) -> None:
    """Exit early if a completed log for the same experiment already exists.

    Args:
        exp_name: Experiment name to check against existing log files.
        log_save_dir: Directory that stores experiment log files.

    Returns:
        None.

    """
    if not log_save_dir.exists():
        return None

    log_path = log_save_dir / f"{exp_name}.log"
    if not log_path.exists():
        return None
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if "Experiment completed successfully!" in content:
        sys.exit(0)

    return None


def configure_deterministic_runtime(seed: int, logger: logging.Logger) -> None:
    """Configure deterministic backends without sharing one global RNG stream."""
    configure_deterministic_backend()
    logger.info(
        "Base random seed set to %d; stochastic processes derive independent RNGs.",
        seed,
    )


def init_joint_model(
    config: ConfigParser, logger: logging.Logger, tb_summary_writer: SummaryWriter
) -> JointModel:
    """Construct and move the joint model to the configured device.

    Args:
        config: Parsed experiment configuration.
        logger: Logger instance for model lifecycle messages.
        tb_summary_writer: TensorBoard writer used for training metrics.

    Returns:
        Initialized joint model moved to the configured device.

    """
    base_seed = config.getint("experiment", "seed")
    with torch_seed_scope(
        base_seed,
        "imputer_init",
        config.get("imputer", "model"),
    ) as imputer_seed:
        logger.info("Independent RNG: imputer_init_seed=%d", imputer_seed)
        imputer_model = imputator(config.get("imputer", "model"), config)

    with torch_seed_scope(
        base_seed,
        "predictor_init",
        config.get("predictor", "model"),
    ) as predictor_seed:
        logger.info("Independent RNG: predictor_init_seed=%d", predictor_seed)
        predictor_model = predictor(config.get("predictor", "model"), config)

    joint_model = JointModel(
        imputer=imputer_model,
        predictor=predictor_model,
        optimizer_name=config.get("training", "optimizer"),
        lr=config.getfloat("training", "lr"),
        device=config.get("training", "device"),
        tb_summary_writer=tb_summary_writer,
        logger=logger,
    )
    joint_model.configure_random_generators(base_seed)

    device = config.get("training", "device")
    joint_model.to(device)

    return joint_model


if __name__ == "__main__":
    main()
