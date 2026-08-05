"""FOTSI training/evaluation entry point; the experiment phase is config-driven."""

import hashlib
import json
import logging
import math
import re
import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

ROOT_PATH = Path(__file__).resolve().parent
GLOBAL_CONFIG_PATH = ROOT_PATH / "configs" / "global.ini"

_NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def normalize_config_value(value: str) -> str:
    """Normalize config values so equivalent CLI spellings hash identically."""
    stripped = str(value).strip()
    lower = stripped.lower()
    if lower in {"true", "false"}:
        return lower
    if _NUMERIC_RE.fullmatch(stripped):
        return format(float(stripped), ".12g")
    return stripped


def safe_name_part(value: str) -> str:
    normalized = normalize_config_value(value)
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "-", normalized).strip("-")
    return safe or "none"


def selected_config_payload(config: ConfigParser) -> dict[str, str]:
    exp_type = config.get("experiment", "type")
    payload: dict[str, str] = {}

    def add(section: str, key: str) -> None:
        if config.has_option(section, key):
            payload[f"{section}.{key}"] = normalize_config_value(config.get(section, key))

    add("experiment", "type")
    add("experiment", "seed")

    for section in ("data", "dataset", "imputer", "predictor"):
        if not config.has_section(section):
            continue
        for key in sorted(config.options(section)):
            if section == "data" and key == "d_feature":
                continue
            add(section, key)

    for key in ("optimizer", "lr", "max_epochs", "patience", "min_delta"):
        add("training", key)

    # ``batch_size`` defines the effective optimization batch and therefore
    # the experiment.  ``auto_batch`` and ``micro_batch`` only choose how that
    # effective batch is executed in memory; changing either must not create a
    # new scientific experiment identity.
    for key in ("batch_size",):
        add("loader", key)

    if exp_type in {"B2", "B3", "B4", "B5"}:
        for key in ("subspace_rank", "epsilon", "confidence_delta"):
            add("tip", key)

    return payload


def generate_experiment_name(config: ConfigParser) -> str:
    exp_type = config.get("experiment", "type")
    lr = normalize_config_value(config.get("training", "lr"))
    batch_size = normalize_config_value(config.get("loader", "batch_size"))
    parts = [
        exp_type,
        config.get("data", "dataset"),
        config.get("imputer", "model"),
        config.get("predictor", "model"),
        f"lr{lr}",
        f"bs{batch_size}",
    ]
    if exp_type in {"B2", "B3", "B4", "B5"}:
        rank = normalize_config_value(config.get("tip", "subspace_rank"))
        epsilon = normalize_config_value(config.get("tip", "epsilon"))
        confidence_delta = normalize_config_value(config.get("tip", "confidence_delta"))
        parts.extend([f"rank{rank}", f"eps{epsilon}", f"delta{confidence_delta}"])

    prefix = "_".join(safe_name_part(part) for part in parts)
    payload = selected_config_payload(config)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"{prefix}__{digest}"


def set_generated_experiment_name(
    config: ConfigParser, init_logs: list[tuple[str, str]]
) -> None:
    configured = (
        config.get("experiment", "name").strip()
        if config.has_option("experiment", "name")
        else ""
    )
    generated = generate_experiment_name(config)
    if configured and configured != "default" and configured != generated:
        init_logs.append(
            (
                "warning",
                f"Ignoring configured experiment.name={configured}; using generated name.",
            )
        )
    config.set("experiment", "name", generated)
    init_logs.append(("info", f"Generated experiment name: {generated}"))


def config_from_overrides(overrides: list[str]) -> ConfigParser:
    config, _ = parse_config([], cli_args=overrides)
    return config


def resolve_experiment_name(overrides: list[str]) -> str:
    return config_from_overrides(overrides).get("experiment", "name")


def parse_column_index(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "all":
        return normalized
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"dataset.column_index must be an integer or 'all', got {value!r}"
        ) from exc


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
    manual_micro_batch = config.getint("loader", "micro_batch")
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
    effective_batch_size = config.getint("loader", "batch_size")
    device = config.get("training", "device")
    num_workers = config.getint("loader", "num_workers")
    pin_memory = (
        config.getboolean("loader", "pin_memory")
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
    config, logger, tb_summary_writer = init_exp()

    log_system_info(logger)
    log_config(config, logger)

    seed = int(config.get("experiment", "seed"))
    configure_deterministic_runtime(seed, logger)

    dataset_folder = preprocess_dataset(
        source=config.get("dataset", "source"),
        repo_name=config.get("dataset", "repo_name"),
        dataset_name=config.get("dataset", "dataset_name"),
        frequency=config.get("dataset", "frequency"),
        item_id=config.get("dataset", "item_id"),
        column_index=parse_column_index(config.get("dataset", "column_index")),
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

    load_ckpt = config.get("experiment", "load_ckpt")
    if load_ckpt:
        logger.info("Loading imputer checkpoint from %s", load_ckpt)
        joint_model.imputer.load_state_dict(torch.load(load_ckpt, map_location=config.get("training", "device")))
    else:
        joint_model = run_experiment(
            exp_type=config["experiment"]["type"],
            joint_model=joint_model,
            train_loader=train_loader,
            config=config,
            logger=logger,
        )
    logger.info("Training completed.")

    exp_name = config.get("experiment", "name")
    ckpt_path = Path(config.get("paths", "cache_dir")) / "ckpt" / f"{exp_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(joint_model.imputer.state_dict(), ckpt_path)
    logger.info("Saved checkpoint to %s", ckpt_path)

    train_eval_loader = train_loader.get("train_eval")
    if train_eval_loader is not None:
        with torch_seed_scope(seed, "train_evaluation"):
            evaluate(
                config,
                joint_model,
                train_eval_loader,
                no_gt=False,
                logger=logger,
                split_name="train",
            )

    val_loader = train_loader.get("val")
    if val_loader is not None:
        with torch_seed_scope(seed, "validation_evaluation"):
            val_no_gt = getattr(val_loader, "no_gt", False)
            evaluate(config, joint_model, val_loader, val_no_gt, logger, split_name="val")

    split_no_gt = getattr(eval_loader, "no_gt", False)
    with torch_seed_scope(seed, "test_evaluation"):
        evaluate(
            config,
            joint_model,
            eval_loader,
            split_no_gt,
            logger,
            split_name="test",
        )

    logger.info("Experiment completed successfully!")


def init_exp() -> tuple[ConfigParser, logging.Logger, SummaryWriter]:
    config, init_logs = parse_config([])

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
    cli_args: list[str] | None = None,
) -> tuple[ConfigParser, list[tuple[str, str]]]:
    config = ConfigParser(interpolation=ExtendedInterpolation())

    if not GLOBAL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Global config file not found: {GLOBAL_CONFIG_PATH}")
    config.read(GLOBAL_CONFIG_PATH)

    unknown_args = sys.argv[1:] if cli_args is None else list(cli_args)
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
                if i + 1 >= len(unknown_args) or unknown_args[i + 1].startswith("--"):
                    raise ValueError(f"Missing value for argument {unknown_args[i]}")
                value = unknown_args[i + 1]
                if section == "experiment" and key == "name":
                    init_logs.append(
                        (
                            "warning",
                            "Ignoring --experiment.name; experiment names are generated from config.",
                        )
                    )
                    continue
                if not config.has_section(section):
                    raise ValueError(f"Section {section} not found")
                if not config.has_option(section, key):
                    raise ValueError(f"Key {key} not found in section {section}")
                config.set(section, key, value)
                init_logs.append(
                    ("info", f"Overriding config: [{section}] {key} = {value}")
                )

    for section in ["imputer", "predictor"]:
        model = config.get(section, "model")
        template_path = ROOT_PATH / "configs" / "template" / section / f"{model}.ini"
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
    template_path = ROOT_PATH / "configs" / "template" / "dataset" / f"{dataset_key}.ini"
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

    set_generated_experiment_name(config, init_logs)
    return config, init_logs


def exit_if_experiment_completed(exp_name: str, log_save_dir: Path) -> None:
    if not log_save_dir.exists():
        return

    log_path = log_save_dir / f"{exp_name}.log"
    if not log_path.exists():
        return
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if "Experiment completed successfully!" in content:
        sys.exit(0)


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
