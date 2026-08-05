#!/usr/bin/env python3
"""Compare full-parameter and local-subspace TIP estimators at equal cost.

The script is read-only: it loads an existing B1 checkpoint and evaluates
independent zero-order estimates on validation batches.  Recent imputation
gradients form an orthonormal local update subspace that always contains the
current gradient.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from configparser import ConfigParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import config_from_overrides, parse_column_index  # noqa: E402
from src.data_loader import get_tensor_data_loader, load_h5, preprocess_dataset  # noqa: E402
from src.models.imputators import imputator  # noqa: E402
from src.models.predictors import predictor  # noqa: E402
from src.rng import configure_deterministic_backend, torch_generator, torch_seed_scope  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1_H_M")
    parser.add_argument("--imputer", default="SAITS")
    parser.add_argument("--predictor", default="Chronos")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.max_batches, args.rank, args.k, args.repeats) < 1:
        parser.error("batch, rank, k and repeat counts must be positive")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    return args


def read_best_config(dataset: str, imputer_name: str) -> tuple[Path, str, int]:
    path = ROOT / "configs" / "best_cfg" / f"best_{dataset}_{imputer_name}.ini"
    if not path.exists():
        raise FileNotFoundError(f"Best config not found: {path}")
    cfg = ConfigParser()
    cfg.read(path)
    return (
        Path(cfg.get("paths", "ckpt")),
        cfg.get("training", "lr"),
        cfg.getint("loader", "batch_size"),
    )


def flatten_grad(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([
        (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters()
    ])


def orthonormal_basis(
    vectors: list[torch.Tensor], max_rank: int, relative_tol: float = 1e-6
) -> torch.Tensor:
    """Modified Gram-Schmidt, preserving the current vector as the first axis."""
    basis: list[torch.Tensor] = []
    for vector in vectors:
        residual = vector.detach().clone()
        original_norm = torch.linalg.vector_norm(residual)
        if original_norm <= 1e-12:
            continue
        # Re-orthogonalize once to reduce loss of orthogonality.
        for _ in range(2):
            for axis in basis:
                residual.sub_(axis, alpha=torch.dot(axis, residual))
        norm = torch.linalg.vector_norm(residual)
        if norm / original_norm > relative_tol:
            basis.append(residual / norm)
        if len(basis) >= max_rank:
            break
    if not basis:
        raise RuntimeError("Cannot form a subspace from zero gradients")
    return torch.stack(basis, dim=1)


@torch.no_grad()
def estimate(
    *, imputer_model, pred_model, batch, epsilon: float, k: int,
    generator: torch.Generator, basis: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(imputer_model.parameters()).device
    X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
        tensor.to(device) for tensor in batch
    )
    params = list(imputer_model.parameters())
    base = torch.cat([p.detach().flatten() for p in params])
    current_grad = flatten_grad(imputer_model)
    directions: list[torch.Tensor] = []
    imputed_pairs: list[torch.Tensor] = []

    def assign(flat: torch.Tensor) -> None:
        offset = 0
        for parameter in params:
            count = parameter.numel()
            parameter.copy_(flat[offset : offset + count].view_as(parameter))
            offset += count

    modes = {module: module.training for module in imputer_model.modules()}
    try:
        imputer_model.eval()
        for _ in range(k):
            if basis is None:
                direction = torch.randn(
                    base.shape, device=device, dtype=base.dtype, generator=generator
                )
            else:
                coordinate = torch.randn(
                    basis.shape[1], device=device, dtype=base.dtype,
                    generator=generator,
                )
                direction = basis @ coordinate
            directions.append(direction)
            assign(base + epsilon * direction)
            imputed_pairs.append(
                imputer_model(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
            )
            assign(base - epsilon * direction)
            imputed_pairs.append(
                imputer_model(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
            )

        predictions = pred_model(torch.cat(imputed_pairs))["pred_data"]
        batch_size = X_NA.shape[0]
        mask = 1.0 - Y_N_mask
        denominator = mask.sum() + 1e-8
        estimate_flat = torch.zeros_like(base)
        alignments: list[torch.Tensor] = []
        for index, direction in enumerate(directions):
            plus = predictions[
                2 * index * batch_size : (2 * index + 1) * batch_size
            ]
            minus = predictions[
                (2 * index + 1) * batch_size : (2 * index + 2) * batch_size
            ]
            q_plus = ((plus - Y_N).square() * mask).sum() / denominator
            q_minus = ((minus - Y_N).square() * mask).sum() / denominator
            coefficient = (q_plus - q_minus) / (2.0 * epsilon)
            contribution = coefficient * direction
            estimate_flat.add_(contribution)
            alignments.append(torch.dot(current_grad, contribution))
        return estimate_flat / k, torch.stack(alignments)
    finally:
        assign(base)
        for module, training in modes.items():
            module.train(training)


def pairwise_cosine(estimates: list[torch.Tensor]) -> float:
    values = []
    for left in range(len(estimates)):
        for right in range(left + 1, len(estimates)):
            values.append(torch.nn.functional.cosine_similarity(
                estimates[left], estimates[right], dim=0, eps=1e-12
            ).item())
    return sum(values) / len(values) if values else float("nan")


def sign_agreement(values: list[float]) -> float:
    negative = sum(value < 0 for value in values)
    return max(negative, len(values) - negative) / len(values)


def summarize_method(
    estimates: list[torch.Tensor], alignments: list[torch.Tensor], grad: torch.Tensor
) -> dict[str, float]:
    dots = [torch.dot(grad, estimate).item() for estimate in estimates]
    correction_ratios = []
    cosines = []
    for estimate, dot in zip(estimates, dots, strict=True):
        gp_norm_sq = torch.dot(estimate, estimate).clamp_min(1e-12)
        correction = (dot / gp_norm_sq) * estimate
        correction_ratios.append(
            (torch.linalg.vector_norm(correction)
             / torch.linalg.vector_norm(grad).clamp_min(1e-12)).item()
        )
        cosines.append(
            dot / (
                torch.linalg.vector_norm(grad).item()
                * torch.linalg.vector_norm(estimate).item() + 1e-12
            )
        )
    sample_tensor = torch.cat(alignments)
    return {
        "repeat_cosine": pairwise_cosine(estimates),
        "conflict_sign_agreement": sign_agreement(dots),
        "dot_mean": sum(dots) / len(dots),
        "dot_std": torch.tensor(dots, dtype=torch.float64).std(
            unbiased=len(dots) > 1
        ).item(),
        "cosine_abs_mean": sum(abs(value) for value in cosines) / len(cosines),
        "candidate_correction_ratio_mean": sum(correction_ratios) / len(correction_ratios),
        "directional_sample_std": sample_tensor.std(
            unbiased=sample_tensor.numel() > 1
        ).item(),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("tip_subspace")
    configure_deterministic_backend()
    default_ckpt, lr, default_batch_size = read_best_config(args.dataset, args.imputer)
    checkpoint = (args.checkpoint or default_ckpt).expanduser().resolve()
    batch_size = args.batch_size or default_batch_size
    overrides = [
        "--data.dataset", args.dataset,
        "--imputer.model", args.imputer,
        "--predictor.model", args.predictor,
        "--training.lr", lr,
        "--loader.batch_size", str(batch_size),
        "--experiment.seed", str(args.seed),
    ]
    if args.device:
        overrides.extend(["--training.device", args.device])
    config = config_from_overrides(overrides)
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
        random_seed=args.seed,
        logger=logger,
        save_path=Path(config.get("paths", "cache_dir")) / "data",
    )
    dataset = load_h5(dataset_folder / "dataset.h5")
    config.set("data", "d_feature", str(dataset["train"]["X_N"].shape[2]))
    config.set("data", "lookback", str(dataset["train"]["X_N"].shape[1]))
    val_loader = get_tensor_data_loader(dataset, batch_size=batch_size)[0]["val"]
    device = config.get("training", "device")
    with torch_seed_scope(args.seed, "tip_subspace_imputer", args.imputer):
        imputer_model = imputator(args.imputer, config).to(device)
    imputer_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    pred_model = predictor(args.predictor, config).to(device).eval()

    history: list[torch.Tensor] = []
    batch_results = []
    print(f"Checkpoint: {checkpoint}")
    print(
        f"Equal budget: k={args.k} ({2 * args.k} queries/estimate), "
        f"rank<={args.rank}, repeats={args.repeats}, epsilon={args.epsilon:g}"
    )
    for batch_index, batch in enumerate(val_loader):
        if batch_index >= args.max_batches:
            break
        tensors = [tensor.to(device) for tensor in batch]
        imputer_model.train()
        imputer_model.zero_grad(set_to_none=True)
        output = imputer_model(tensors[1], tensors[2], tensors[0], tensors[3])
        output["imp_loss"].backward()
        current_grad = flatten_grad(imputer_model)
        basis = orthonormal_basis([current_grad, *reversed(history)], args.rank)
        history.append(current_grad.detach().clone())
        history_limit = max(0, args.rank - 1)
        history = history[-history_limit:] if history_limit else []

        full_estimates, full_samples = [], []
        sub_estimates, sub_samples = [], []
        for repeat in range(args.repeats):
            full_rng = torch_generator(
                args.seed, "tip_subspace_full", batch_index, repeat, device=device
            )
            sub_rng = torch_generator(
                args.seed, "tip_subspace_local", batch_index, repeat, device=device
            )
            full, samples = estimate(
                imputer_model=imputer_model, pred_model=pred_model, batch=batch,
                epsilon=args.epsilon, k=args.k, generator=full_rng, basis=None,
            )
            local, local_samples = estimate(
                imputer_model=imputer_model, pred_model=pred_model, batch=batch,
                epsilon=args.epsilon, k=args.k, generator=sub_rng, basis=basis,
            )
            full_estimates.append(full)
            full_samples.append(samples)
            sub_estimates.append(local)
            sub_samples.append(local_samples)

        result = {
            "batch": batch_index,
            "effective_rank": basis.shape[1],
            "full": summarize_method(full_estimates, full_samples, current_grad),
            "subspace": summarize_method(sub_estimates, sub_samples, current_grad),
        }
        batch_results.append(result)
        print(f"\nBatch {batch_index + 1}: effective_rank={basis.shape[1]}")
        for name in ("full", "subspace"):
            values = result[name]
            print(
                f"  {name:8s} repeat_cos={values['repeat_cosine']:+.4f} | "
                f"sign_agree={values['conflict_sign_agreement']:.3f} | "
                f"|cos(gI,gP)|={values['cosine_abs_mean']:.4f} | "
                f"correction={values['candidate_correction_ratio_mean']:.4f}"
            )

    summary = {"config": vars(args), "batches": batch_results}
    for name in ("full", "subspace"):
        summary[name] = {
            key: sum(row[name][key] for row in batch_results) / len(batch_results)
            for key in batch_results[0][name]
        }
    print("\nMean across batches")
    for name in ("full", "subspace"):
        values = summary[name]
        print(
            f"  {name:8s} repeat_cos={values['repeat_cosine']:+.4f} | "
            f"sign_agree={values['conflict_sign_agreement']:.3f} | "
            f"|cos(gI,gP)|={values['cosine_abs_mean']:.4f} | "
            f"correction={values['candidate_correction_ratio_mean']:.4f}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(json.dumps(summary, default=str))
        args.json.write_text(json.dumps(serializable, indent=2) + "\n")
        print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
