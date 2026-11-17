"""Complete-series preprocessing and chronological window construction."""

import logging
from pathlib import Path

import numpy as np

from src.rng import numpy_rng

_logger = logging.getLogger(__name__)
TIME_REVISION = "83e3d0b3be28d11c7182bffcc1892d19b36c4da1"


def _snapshot_subtree(allow_patterns: str) -> str:
    """The single directory ``allow_patterns`` selects, as a relative path."""
    return allow_patterns.split("/*", 1)[0].strip("/")


def _require_subtree(snapshot_dir: str, allow_patterns: str) -> str:
    """Verify that a snapshot contains the requested dataset directory."""
    subtree = _snapshot_subtree(allow_patterns)
    resolved = Path(snapshot_dir) / subtree
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"snapshot {snapshot_dir} does not contain {subtree!r} "
            f"(allow_patterns={allow_patterns!r})"
        )
    return snapshot_dir


def _cached_snapshot(
    repo_id: str, allow_patterns: str, cache_dir=None, local_files_only=False
) -> str:
    """Resolve a pinned dataset snapshot and verify the requested subtree.

    If the Hub is unavailable, use a complete copy from the local cache.
    """
    from huggingface_hub import snapshot_download

    revision = TIME_REVISION if repo_id == "Real-TSF/TIME" else "main"
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return _require_subtree(snapshot_dir, allow_patterns)
    except Exception as error:
        _logger.warning(
            "snapshot %s could not be fetched from the Hub (%s); falling back "
            "to the local cache",
            repo_id,
            error,
        )
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
        local_files_only=True,
        cache_dir=cache_dir,
    )
    return _require_subtree(snapshot_dir, allow_patterns)


def preprocess_dataset(
    series: np.ndarray,
    missing_rate: float,
    artificial_missing_rate: float,
    block_size: int,
    lookback_len: int,
    pred_len: int,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    logger: logging.Logger,
) -> dict[str, dict[str, np.ndarray]]:
    """Prepare the complete selected series with disjoint chronological splits."""
    logger.info("Using complete series: %d points", len(series))
    return _chronological_split(
        series,
        missing_rate,
        artificial_missing_rate,
        block_size,
        lookback_len,
        pred_len,
        val_ratio,
        test_ratio,
        random_seed,
        logger,
    )


def load_raw_series(
    *,
    source: str,
    repo_name: str,
    dataset_name: str,
    frequency: str,
    item_id: str,
    column_index: int | str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> np.ndarray:
    """Load one univariate series from a supported Hugging Face layout."""
    if source == "huggingface":
        from datasets import load_from_disk

        snapshot_dir = _cached_snapshot(
            repo_name, f"{dataset_name}/{frequency}/*", cache_dir, local_files_only
        )
        dataset = load_from_disk(f"{snapshot_dir}/{dataset_name}/{frequency}")
        try:
            row = list(dataset["item_id"]).index(item_id)
        except ValueError as error:
            raise ValueError(
                f"item_id {item_id!r} not found in {dataset_name}"
            ) from error
        target = np.asarray(dataset[row]["target"])
        if column_index == "all":
            series = target.T if target.ndim > 1 else target.reshape(-1, 1)
            return np.asarray(series, dtype=np.float64)
        series = target[column_index, :] if target.ndim > 1 else target
        return np.asarray(series, dtype=np.float64).reshape(-1, 1)

    raise ValueError(f"Unsupported dataset source {source!r}. Expected 'huggingface'.")


def _sample_block_mask(
    rng: np.random.Generator,
    span: int,
    n_cols: int,
    block_size: int,
    rate: float,
) -> np.ndarray:
    """Sample a boolean block-missing mask of length ``span``.

    The span is tiled into consecutive blocks of ``block_size`` points and each
    block is dropped entirely, per column, with probability ``rate``; the last
    block is truncated when ``span`` is not a multiple of ``block_size``.
    """
    block_count = int(np.ceil(span / block_size))
    block_flags = rng.random((block_count, n_cols)) < rate
    return np.repeat(block_flags, block_size, axis=0)[:span]


def _chronological_split(
    series: np.ndarray,
    missing_rate: float,
    artificial_missing_rate: float,
    block_size: int,
    lookback_len: int,
    pred_len: int,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    logger: logging.Logger,
) -> dict[str, dict[str, np.ndarray]]:
    """Allocate disjoint raw ranges in the requested *window-count* ratio.

    Primary synthetic corruption affects contexts. Originally observed future
    values remain offline labels even when that timestamp is hidden in a context.
    Each split and mask role has its own random stream. Reconstruction removes
    further observed context entries independently with artificial_missing_rate.
    """
    if lookback_len < 1 or pred_len < 1 or block_size < 1:
        raise ValueError("lookback, horizon and block size must be positive")
    if not (0 <= val_ratio < 1 and 0 < test_ratio < 1 - val_ratio):
        raise ValueError("Invalid validation and test ratios")
    if not (0 <= missing_rate <= 1 and 0 <= artificial_missing_rate <= 1):
        raise ValueError("Missing rates must be between zero and one")
    if series.ndim != 2 or not series.shape[1] or np.isinf(series).any():
        raise ValueError("Expected a (time, features) series of finite values or NaNs")

    overhead = lookback_len + pred_len - 1
    n_windows = len(series) - (3 if val_ratio else 2) * overhead
    n_train = int(n_windows * (1 - val_ratio - test_ratio))
    n_val = int(n_windows * val_ratio)
    counts = {"train": n_train}
    if val_ratio:
        counts["val"] = n_val
    counts["test"] = n_windows - n_train - n_val
    if min(counts.values()) < 1:
        raise ValueError(
            f"Series of length {len(series)} gives empty split(s): {counts}"
        )

    splits = {}
    start = 0
    for name, count in counts.items():
        end = start + overhead + count
        values = series[start:end]
        absent = np.isnan(values)
        synthetic = _sample_block_mask(
            numpy_rng(random_seed, name, "primary_blocks"),
            len(values),
            values.shape[1],
            block_size,
            missing_rate,
        )
        primary = absent | synthetic
        auxiliary = (
            numpy_rng(random_seed, name, "auxiliary_points").random(values.shape)
            < artificial_missing_rate
        ) & ~primary
        if name == "train":
            if not (~primary).any(axis=0).all():
                raise ValueError(
                    "A training feature has no observed values for standardization"
                )
            observed = np.where(primary, np.nan, values)
            mean = np.nanmean(observed, axis=0)
            scale = np.nanstd(observed, axis=0)
            scale = np.where(scale > 0, scale, 1.0)
        z = np.nan_to_num((values - mean) / scale, nan=0).astype(np.float32)
        visible = np.where(primary, 0, z)
        masked = np.where(auxiliary, 0, visible)
        context = slice(None, -pred_len)
        future = slice(lookback_len, None)
        split = {}
        for key, arr in (
            ("X_N", visible),
            ("X_N_mask", primary),
            ("X_NA", masked),
            ("X_NA_mask", primary | auxiliary),
            ("X_A_mask", auxiliary),
        ):
            split[key] = _sliding_windows(arr[context], lookback_len)
        # Complete contexts provide held-out imputation targets.
        split["X"] = _sliding_windows(z[context], lookback_len)
        split["X_eval_mask"] = _sliding_windows(
            (synthetic & ~absent)[context], lookback_len
        )
        split["Y_N"] = _sliding_windows(z[future], pred_len)
        split["Y_N_mask"] = _sliding_windows(absent[future], pred_len)
        labelled = (~split["Y_N_mask"]).any(axis=(1, 2))
        if not labelled.any():
            raise ValueError(f"No labelled forecast windows in {name} split")
        if not labelled.all():
            split = {key: value[labelled] for key, value in split.items()}
        splits[name] = split
        logger.info(
            "%s raw [%d,%d), windows=%d, labelled=%d, primary_missing=%.4f, "
            "artificial_given_observed=%.4f",
            name,
            start,
            end,
            count,
            int(labelled.sum()),
            primary.mean(),
            auxiliary.sum() / max((~primary).sum(), 1),
        )
        start = end
    return splits


def _sliding_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, window_size, axis=0)
    return np.ascontiguousarray(np.moveaxis(windows, -1, 1))
