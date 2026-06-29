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
import random
import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import numpy as np
import torch
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

GLOBAL_CONFIG_PATH = Path("./configs/global.ini")


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

    seed = int(config.get("experiment", "random_seed"))
    set_random_seed(seed)
    logger.info(f"Random seed set to {seed}")

    dataset_folder = preprocess_dataset(
        dataset_name=config.get("dataset", "dataset_name"),
        frequency=config.get("dataset", "frequency"),
        item_id=config.get("dataset", "item_id"),
        column_index=config.getint("dataset", "column_index"),
        missing_rate=config.getfloat("dataset", "missing_rate"),
        artificial_missing_rate=config.getfloat("dataset", "artificial_missing_rate"),
        block_size=config.getint("dataset", "block_size"),
        lookback_len=config.getint("dataset", "lookback_len"),
        pred_len=config.getint("dataset", "pred_len"),
        val_ratio=config.getfloat("dataset", "val_ratio"),
        test_ratio=config.getfloat("dataset", "test_ratio"),
        random_seed=config.getint("experiment", "random_seed"),
        logger=logger,
        save_path=Path(config.get("paths", "preprocessed_data_dir")),
    )

    dataset = load_h5(dataset_folder / "dataset.h5")

    config.set("dataset", "d_feature", str(dataset["train"]["X_N"].shape[2]))
    config.set("dataset", "lookback_len", str(dataset["train"]["X_N"].shape[1]))

    target_batch_size = config.getint("training", "batch_size")
    device = config.get("training", "device")
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
    loader_kwargs = {
        "num_workers": dataloader_num_workers,
        "pin_memory": pin_memory,
    }
    auto_batch = config.has_option("training", "auto_batch_size") and config.getboolean(
        "training", "auto_batch_size"
    )
    manual_micro_batch = config.getint("training", "micro_batch_size", fallback=0)

    joint_model = init_joint_model(config, logger, tb_summary_writer)

    if manual_micro_batch > 0:
        micro_batch_size = manual_micro_batch
        accum_steps = math.ceil(target_batch_size / micro_batch_size)
        logger.info(
            f"Using manual micro_batch_size={micro_batch_size} "
            f"(target={target_batch_size}, accum_steps={accum_steps})"
        )
    elif auto_batch and target_batch_size > 1:
        sample_loader, _, _ = get_tensor_data_loader(
            dataset,
            batch_size=target_batch_size,
            **loader_kwargs,
        )
        sample_batch = next(iter(sample_loader["train"]))

        micro_batch_size = find_max_batch_size(
            joint_model,
            sample_batch,
            target_batch_size,
            config.get("training", "device"),
            logger,
            config,
        )
        accum_steps = math.ceil(target_batch_size / micro_batch_size)
        logger.info(
            f"Effective batch_size: {target_batch_size} | "
            f"Micro batch_size: {micro_batch_size} | "
            f"Gradient accumulation steps: {accum_steps}"
        )
    else:
        micro_batch_size = target_batch_size
        accum_steps = 1

    joint_model.accum_steps = accum_steps
    joint_model.micro_batch_size = micro_batch_size

    train_loader, eval_loader, no_gt = get_tensor_data_loader(
        dataset,
        batch_size=micro_batch_size,
        **loader_kwargs,
    )

    joint_model = run_experiment(
        exp_type=config["experiment"]["exp_type"],
        joint_model=joint_model,
        train_loader=train_loader,
        config=config,
        logger=logger,
    )
    logger.info("Training completed.")

    evaluate(config, joint_model, eval_loader, no_gt, logger)
    joint_model.delete_checkpoint()

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

    exp_name = config.get("experiment", "exp_name")
    exit_if_experiment_completed(
        exp_name=exp_name,
        log_save_dir=Path(config.get("paths", "log_save_dir")),
    )

    log_save_dir = Path(config.get("paths", "log_save_dir"))
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

    for section in ["imputator", "predictor"]:
        model_type = config.get(section, "model_type")
        template_path = Path(f"./configs/template/{section}/{model_type}.ini")
        if template_path.exists():
            init_logs.append(
                (
                    "info",
                    f"Loading template config for {model_type} from {template_path}",
                )
            )
            config.read(template_path)
        else:
            raise FileNotFoundError(
                f"Template config for '{model_type}' not found at {template_path}"
            )

    dataset_key = config.get("dataset", "dataset")
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


def set_random_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Random seed used by Python, NumPy, and PyTorch.

    Returns:
        None.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    joint_model = JointModel(
        imputer=imputator(config.get("imputator", "model_type"), config),
        predictor=predictor(config.get("predictor", "model_type"), config),
        model_save_dir=Path(config.get("paths", "ckpt_save_dir"))
        / config.get("experiment", "exp_name"),
        optimizer_name=config.get("training", "optimizer"),
        lr=config.getfloat("training", "lr"),
        device=config.get("training", "device"),
        tb_summary_writer=tb_summary_writer,
        logger=logger,
    )

    device = config.get("training", "device")
    joint_model.to(device)

    return joint_model


if __name__ == "__main__":
    main()
