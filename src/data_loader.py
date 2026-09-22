"""Data preprocessing and dataloader construction utilities."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, TensorDataset

from src.rng import numpy_rng

_logger = logging.getLogger(__name__)

# Keep dataset size fixed as an internal preprocessing policy rather than an
# experiment parameter that can accidentally enter a sweep.
_MAX_EXPERIMENT_SERIES_POINTS = 2_000


def _snapshot_subtree(allow_patterns: str) -> str:
    """The single directory ``allow_patterns`` selects, as a relative path."""
    return allow_patterns.split("/*", 1)[0].strip("/")


def _require_subtree(snapshot_dir: str, allow_patterns: str) -> str:
    """Fail loudly when a resolved snapshot does not hold the requested files.

    A snapshot directory that resolves successfully says nothing about its
    contents, so the subtree is checked explicitly before it is handed to
    ``load_from_disk``.  Without this the caller dies on a bare
    ``FileNotFoundError`` naming a directory that was simply never downloaded.
    """
    subtree = _snapshot_subtree(allow_patterns)
    resolved = Path(snapshot_dir) / subtree
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"snapshot {snapshot_dir} does not contain {subtree!r} "
            f"(allow_patterns={allow_patterns!r})"
        )
    return snapshot_dir


def _cached_snapshot(repo_id: str, allow_patterns: str) -> str:
    """Resolve a dataset snapshot, asking the Hub on every call.

    The Hub is authoritative here.  An earlier version tried the local cache
    first and fell back to the network only on a miss, but cache resolution
    does not consider ``allow_patterns``: once the revision is cached,
    huggingface_hub hands back the snapshot directory even when not one file
    matched, so a machine holding an incomplete snapshot never fell through to
    the network and every run on a missing dataset failed with a bare
    ``FileNotFoundError``.  The directory list is now always re-resolved, and
    the requested subtree is verified before the path is returned.

    Only when the Hub itself is unreachable does a locally cached snapshot
    serve the request -- and then only after the same subtree check, so a
    partial cache fails loudly instead of silently.
    """
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            allow_patterns=allow_patterns,
        )
        return _require_subtree(snapshot_dir, allow_patterns)
    except Exception as error:  # noqa: BLE001 - any failure means "try the cache"
        _logger.warning(
            "snapshot %s could not be fetched from the Hub (%s); falling back "
            "to the local cache",
            repo_id,
            error,
        )
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        allow_patterns=allow_patterns,
        local_files_only=True,
    )
    return _require_subtree(snapshot_dir, allow_patterns)


def preprocess_dataset(
    source: str,
    repo_name: str,
    dataset_name: str,
    frequency: str,
    item_id: str,
    column_index: int | str,
    missing_rate: float,
    artificial_missing_rate: float,
    block_size: int,
    lookback_len: int,
    pred_len: int,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    logger: logging.Logger,
    split_mode: str = "chronological",
) -> dict[str, dict[str, Any]]:
    """Preprocess raw data and return the windowed tensors.

    Nothing is written to disk.  The caller's Stage returns this dictionary, and
    expman's snapshot of that return value is what every other run reuses; a
    second store here would only duplicate it.

    Args:
        source: Dataset storage format (``huggingface`` or
            ``huggingface_table``).
        repo_name: Hugging Face dataset repository name.
        dataset_name: Dataset name or configuration in the repository.
        frequency: Sampling frequency subfolder under the dataset.
        item_id: Item index within the dataset.
        column_index: Target column index, or ``"all"`` for all variables.
        missing_rate: Rate used to inject natural missing blocks when needed.
        artificial_missing_rate: Rate used to inject artificial missing blocks.
        block_size: Block size used for missing-block simulation.
        lookback_len: Input window length.
        pred_len: Forecast horizon length.
        val_ratio: Validation split ratio (0 to disable).
        test_ratio: Test split ratio.
        random_seed: Random seed for reproducible preprocessing.
        logger: Logger used to report preprocessing progress.
        split_mode: ``"chronological"`` (default) cuts the series into
            contiguous train/validation/test ranges; ``"shuffled"`` winds the
            whole series first and assigns windows to the three splits at
            random, holding the 7:1:2 sample counts.

    Returns:
        ``{"train": ..., "val": ..., "test": ...}``, the split dictionaries
        :func:`get_tensor_data_loader` consumes.

    """
    data = load_raw_series(
        source=source,
        repo_name=repo_name,
        dataset_name=dataset_name,
        frequency=frequency,
        item_id=item_id,
        column_index=column_index,
    )
    original_length = len(data)
    data = data[:_MAX_EXPERIMENT_SERIES_POINTS]
    logger.info(
        "Using the first %d of %d raw time points for %s",
        len(data), original_length, dataset_name,
    )
    if split_mode == "shuffled":
        train_set, val_set, test_set, _ = _shuffled_split(
            data,
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
        logger.info("Preprocessed data ready")
        return {"train": train_set, "val": val_set, "test": test_set}
    if split_mode != "chronological":
        raise ValueError(
            f"Unknown split_mode {split_mode!r}: expected 'chronological' or "
            "'shuffled'."
        )

    # Conventional ETT boundaries describe the full series and are invalid
    # after applying the common prefix policy.
    ett_boundaries = None

    # A complete dataset is asked to impute block missing where it is measured:
    # the synthetic natural mask is laid down over the test range only, while
    # the artificial mask -- block missing from the same sampler, spread over
    # the whole series -- is what training and validation splits carry.  A
    # dataset that brings its own NaNs keeps them everywhere -- those are real.
    bounds = split_boundaries(
        len(data), lookback_len, pred_len, val_ratio, test_ratio, ett_boundaries
    )
    data, N_data, NA_data, N_mask, A_mask, NA_mask = generate_missing(
        data,
        missing_rate,
        artificial_missing_rate,
        block_size,
        random_seed,
        logger,
        natural_span=(bounds[4], bounds[5]),
    )

    train_data, val_data, test_data = split_data(
        data,
        N_data,
        NA_data,
        N_mask,
        A_mask,
        NA_mask,
        lookback_len,
        pred_len,
        val_ratio,
        test_ratio,
        fixed_boundaries=ett_boundaries,
    )

    train_data, val_data, test_data, scaler = standardize_data(
        train_data, val_data, test_data
    )

    train_data, val_data, test_data = white_nan_with_zero(
        train_data, val_data, test_data
    )

    train_set, val_set, test_set = truncate_data(
        train_data, val_data, test_data, lookback_len, pred_len
    )

    logger.info("Preprocessed data ready")
    return {"train": train_set, "val": val_set, "test": test_set}


def load_raw_series(
    *,
    source: str,
    repo_name: str,
    dataset_name: str,
    frequency: str,
    item_id: str,
    column_index: int | str,
) -> np.ndarray:
    """Load one univariate series from a supported Hugging Face layout."""
    if source == "huggingface":
        snapshot_dir = _cached_snapshot(
            repo_name, f"{dataset_name}/{frequency}/*"
        )
        dataset = load_from_disk(f"{snapshot_dir}/{dataset_name}/{frequency}")
        filtered = dataset.filter(lambda x: x["item_id"] == item_id)
        if len(filtered) == 0:
            raise ValueError(f"item_id '{item_id}' not found in dataset")
        target = np.asarray(filtered[0]["target"])
        if column_index == "all":
            series = target.T if target.ndim > 1 else target.reshape(-1, 1)
            return np.asarray(series, dtype=np.float64)
        series = target[column_index, :] if target.ndim > 1 else target
        return np.asarray(series, dtype=np.float64).reshape(-1, 1)

    if source == "huggingface_table":
        dataset = load_dataset(repo_name, dataset_name, split="train")
        numeric_columns = [
            name for name, feature in dataset.features.items()
            if name != "date" and feature.dtype in {
                "float16", "float32", "float64",
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64",
            }
        ]
        if column_index == "all":
            return np.column_stack(
                [np.asarray(dataset[column], dtype=np.float64) for column in numeric_columns]
            )
        if not 0 <= column_index < len(numeric_columns):
            raise IndexError(
                f"column_index={column_index} is out of range for numeric "
                f"columns {numeric_columns}"
            )
        column = numeric_columns[column_index]
        return np.asarray(dataset[column], dtype=np.float64).reshape(-1, 1)

    raise ValueError(
        f"Unsupported dataset source '{source}'. Expected 'huggingface' or "
        "'huggingface_table'."
    )


def get_ett_split_boundaries(dataset_name: str) -> tuple[int, int, int] | None:
    """Return the conventional 12/4/4-month boundaries for ETT datasets."""
    if dataset_name in {"ETTh1", "ETTh2"}:
        points_per_month = 30 * 24
    elif dataset_name in {"ETTm1", "ETTm2"}:
        points_per_month = 30 * 24 * 4
    else:
        return None
    return (
        12 * points_per_month,
        16 * points_per_month,
        20 * points_per_month,
    )


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
    block is truncated when ``span`` is not a multiple of ``block_size``.  Both
    the natural and the artificial mask are drawn from this one sampler -- they
    differ only in rate and in the span they are laid over.
    """
    block_count = int(np.ceil(span / block_size))
    block_flags = rng.random((block_count, n_cols)) < rate
    return np.repeat(block_flags, block_size, axis=0)[:span]


def generate_missing(
    data: np.ndarray,
    missing_rate: float,
    artificial_missing_rate: float,
    block_size: int,
    random_seed: int,
    logger: logging.Logger,
    natural_span: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inject natural and artificial missing values and return masks.

    Args:
        data: Input time-series array.
        missing_rate: Natural missing rate used when raw data has no NaNs.
        artificial_missing_rate: Artificial missing rate applied to observed
            values as block missing over the whole series.
        block_size: Block size used to sample missing intervals.
        random_seed: Seed for random mask generation.
        logger: Logger for missing-data diagnostics.
        natural_span: Half-open range the synthetic natural blocks are confined
            to.  A complete dataset gets its block missing only where it will be
            measured -- the test split.  ``None`` spreads the blocks over the
            whole series, which is what a dataset with no splits of its own
            still needs.  Natural NaNs already present in the data are never
            moved.

    Returns:
        Tuple containing complete data, natural-missing data, incomplete data, and
        natural/artificial/combined missing masks.

    """
    natural_rng = numpy_rng(random_seed, "missing_mask", "natural")
    artificial_rng = numpy_rng(random_seed, "missing_mask", "artificial")

    natural_mask = np.isnan(data)
    if natural_mask.any():
        logger.info(
            f"Found natural missing values in the dataset. Missing rate: "
            f"{natural_mask.mean():.4f}"
        )
        complete_data = None
        natural_missing_data = data.copy()
    else:
        complete_data = data.copy()
        natural_mask = np.zeros(data.shape, dtype=bool)
        start, end = (0, len(data)) if natural_span is None else natural_span
        natural_mask[start:end] = _sample_block_mask(
            natural_rng, end - start, data.shape[1], block_size, missing_rate
        )
        natural_missing_data = data.copy()
        natural_missing_data[natural_mask] = np.nan

    # Artificial missing is block missing from the same sampler, laid over the
    # whole series so every split carries training supervision; blocks falling
    # on natural missing positions are dropped, keeping the two masks disjoint.
    artificial_mask = _sample_block_mask(
        artificial_rng, len(data), data.shape[1], block_size,
        artificial_missing_rate,
    )
    artificial_mask &= ~natural_mask

    missing_mask = natural_mask | artificial_mask
    incomplete_data = data.copy()
    incomplete_data[missing_mask] = np.nan

    assert natural_missing_data.shape == data.shape
    assert incomplete_data.shape == data.shape
    assert natural_mask.shape == data.shape and natural_mask.dtype == bool
    assert artificial_mask.shape == data.shape and artificial_mask.dtype == bool
    assert missing_mask.shape == data.shape and missing_mask.dtype == bool
    assert not (natural_mask & artificial_mask).any(), \
        "Natural and artificial masks must be disjoint"
    assert (missing_mask == (natural_mask | artificial_mask)).all(), \
        "NA_mask must equal N_mask | A_mask"
    assert np.isnan(natural_missing_data[natural_mask]).all(), \
        "N_data must be NaN where N_mask is True"
    assert np.isnan(incomplete_data[missing_mask]).all(), \
        "NA_data must be NaN where NA_mask is True"
    assert not np.isnan(natural_missing_data[~natural_mask]).any(), \
        "N_data must be finite where N_mask is False"
    if complete_data is not None:
        assert complete_data.shape == data.shape
        assert not np.isnan(complete_data).any(), "complete_data must have no NaN"

    return (
        complete_data,
        natural_missing_data,
        incomplete_data,
        natural_mask,
        artificial_mask,
        missing_mask,
    )


def split_boundaries(
    len_data: int,
    lookback_len: int,
    pred_len: int,
    val_ratio: float,
    test_ratio: float,
    fixed_boundaries: tuple[int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Return the train/val/test slice bounds of one raw series.

    Masking needs the test range before the splits exist, so that a complete
    dataset's synthetic block missing can be placed only where it is measured.
    """
    if fixed_boundaries is not None:
        train_end, val_end, test_end = fixed_boundaries
        if not (0 < train_end < val_end < test_end <= len_data):
            raise ValueError(
                f"Invalid fixed split boundaries {fixed_boundaries} for "
                f"series length {len_data}."
            )
        train_start = 0
        val_start = train_end - lookback_len
        test_start = val_end - lookback_len
        if val_start < 0 or test_start < 0:
            raise ValueError("Lookback is too long for the fixed split boundaries.")
        return train_start, train_end, val_start, val_end, test_start, test_end

    if val_ratio < 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError(
            "Expected val_ratio >= 0, test_ratio > 0, "
            "and val_ratio + test_ratio < 1."
        )

    overhead = lookback_len + pred_len - 1
    min_len = overhead + 1
    num_splits = 2 if val_ratio == 0 else 3
    if len_data < num_splits * min_len:
        raise ValueError(
            f"Time series too short ({len_data}) to create non-overlapping "
            f"splits with lookback={lookback_len}, pred={pred_len}."
        )

    total_windows = len_data - num_splits * overhead
    train_windows = int(total_windows * (1 - val_ratio - test_ratio))
    val_windows = int(total_windows * val_ratio)
    train_raw_len = train_windows + overhead
    val_raw_len = val_windows + overhead if val_ratio > 0 else 0

    train_start = 0
    train_end = train_raw_len
    val_start = train_end
    val_end = val_start + val_raw_len
    return (
        train_start,
        train_end,
        val_start,
        val_end,
        val_end if val_ratio > 0 else train_end,
        len_data,
    )


def split_data(
    data: np.ndarray | None,
    N_data: np.ndarray,
    NA_data: np.ndarray,
    N_mask: np.ndarray,
    A_mask: np.ndarray,
    NA_mask: np.ndarray,
    lookback_len: int,
    pred_len: int,
    val_ratio: float,
    test_ratio: float,
    fixed_boundaries: tuple[int, int, int] | None = None,
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None] | None,
    dict[str, np.ndarray | None],
]:
    """Split raw time series into train, validation, and test blocks.

    ETT datasets use their conventional fixed 12/4/4-month target ranges.
    Their validation and test blocks include the preceding lookback context.
    Other datasets use contiguous, non-overlapping ratio-based blocks with
    compensation for windowing overhead.

    Args:
        data: Optional complete data without natural missing values.
        N_data: Data containing only natural missing values.
        NA_data: Data containing natural and artificial missing values.
        N_mask: Natural missing mask.
        A_mask: Artificial missing mask.
        NA_mask: Combined missing mask.
        lookback_len: Input lookback length.
        pred_len: Prediction length.
        val_ratio: Validation split ratio (applied to window counts).
            Use 0 to disable validation.
        test_ratio: Test split ratio (applied to window counts).
        fixed_boundaries: Optional train/validation/test target end indices.

    Returns:
        Tuple of train, optional validation, and test dictionaries.

    """
    len_data = len(N_data)
    (
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
    ) = split_boundaries(
        len_data, lookback_len, pred_len, val_ratio, test_ratio, fixed_boundaries
    )

    def _make_split(start: int, end: int) -> dict[str, np.ndarray | None]:
        return {
            "N_data": N_data[start:end, :],
            "N_mask": N_mask[start:end, :],
            "NA_data": NA_data[start:end, :],
            "NA_mask": NA_mask[start:end, :],
            "A_mask": A_mask[start:end, :],
            "data": data[start:end, :] if data is not None else None,
        }

    train_split = _make_split(train_start, train_end)
    val_split = _make_split(val_start, val_end) if fixed_boundaries or val_ratio > 0 else None
    test_split = _make_split(test_start, test_end)

    _expected_keys = {"N_data", "N_mask", "NA_data", "NA_mask", "A_mask", "data"}
    assert train_split.keys() == _expected_keys
    assert test_split.keys() == _expected_keys
    assert train_split["N_data"].shape[0] > 0, "Train split must be non-empty"
    assert test_split["N_data"].shape[0] > 0, "Test split must be non-empty"
    if fixed_boundaries is None:
        total_split_len = train_split["N_data"].shape[0] + test_split["N_data"].shape[0]
        if val_split is not None:
            total_split_len += val_split["N_data"].shape[0]
        assert total_split_len == len_data, "Split lengths must sum to original length"

    return train_split, val_split, test_split


class _IgnoreNaNScaler:
    """Column-wise z-score transform whose statistics ignore NaNs.

    NaNs pass through the transform unchanged so downstream NaN
    replacement happens after standardization.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.scale_


def standardize_data(
    train_data: dict[str, np.ndarray | None],
    val_data: dict[str, np.ndarray | None] | None,
    test_data: dict[str, np.ndarray | None],
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None] | None,
    dict[str, np.ndarray | None],
    _IgnoreNaNScaler,
]:
    """Fit a scaler on train data and transform all data partitions.

    Args:
        train_data: Training split dictionary.
        val_data: Optional validation split dictionary.
        test_data: Test split dictionary.

    Returns:
        Scaled train/val/test dictionaries and the fitted scaler.

    """
    # StandardScaler propagates NaN into mean_/scale_ when the series
    # carries natural missing values; fit on observed values instead.
    train_observed = train_data["N_data"]
    mean = np.nanmean(train_observed, axis=0)
    std = np.nanstd(train_observed, axis=0)
    # Constant columns would divide by zero; centering alone keeps them at zero.
    std = np.where(std == 0, 1.0, std)
    scaler = _IgnoreNaNScaler(mean, std)

    def _scale(
        split: dict[str, np.ndarray | None],
    ) -> dict[str, np.ndarray | None] | None:
        if split is None:
            return None
        s = split.copy()
        s["N_data"] = scaler.transform(split["N_data"])
        s["NA_data"] = scaler.transform(split["NA_data"])
        s["data"] = (
            scaler.transform(split["data"]) if split["data"] is not None else None
        )
        return s

    train_data_scaled = _scale(train_data)
    val_data_scaled = _scale(val_data)
    test_data_scaled = _scale(test_data)

    assert not np.any(np.isnan(scaler.mean_)), "Scaler mean must not be NaN"
    assert not np.any(np.isnan(scaler.scale_)), "Scaler scale must not be NaN"
    assert np.all(scaler.scale_ > 0), "Scaler scale must be positive"
    assert np.array_equal(
        np.isnan(train_data_scaled["N_data"]), np.isnan(train_data["N_data"])
    ), "NaN positions must be unchanged after scaling"
    assert np.all(np.isfinite(
        train_data_scaled["N_data"][~np.isnan(train_data["N_data"])]
    )), "Observed positions must be finite after scaling"

    return train_data_scaled, val_data_scaled, test_data_scaled, scaler


def white_nan_with_zero(
    train_data: dict[str, np.ndarray | None],
    val_data: dict[str, np.ndarray | None] | None,
    test_data: dict[str, np.ndarray | None],
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None] | None,
    dict[str, np.ndarray | None],
]:
    """Replace NaN values in model inputs with zeros.

    Args:
        train_data: Training split dictionary.
        val_data: Optional validation split dictionary.
        test_data: Test split dictionary.

    Returns:
        Train, optional val, and test dictionaries with NaNs → 0.

    """
    def _fill(split: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
        if split is None:
            return None
        s = split.copy()
        s["N_data"] = np.nan_to_num(split["N_data"], nan=0.0)
        s["NA_data"] = np.nan_to_num(split["NA_data"], nan=0.0)
        return s

    train_data_filled = _fill(train_data)
    val_data_filled = _fill(val_data)
    test_data_filled = _fill(test_data)

    assert not np.isnan(train_data_filled["N_data"]).any(), \
        "N_data must have no NaN after filling"
    assert not np.isnan(train_data_filled["NA_data"]).any(), \
        "NA_data must have no NaN after filling"
    assert np.allclose(
        train_data_filled["N_data"][~np.isnan(train_data["N_data"])],
        train_data["N_data"][~np.isnan(train_data["N_data"])],
    ), "Observed values must be unchanged after NaN filling"
    assert np.all(
        train_data_filled["N_data"][np.isnan(train_data["N_data"])] == 0.0
    ), "NaN positions must be filled with 0"

    return train_data_filled, val_data_filled, test_data_filled


def truncate_data(
    train_data: dict[str, np.ndarray | None],
    val_data: dict[str, np.ndarray | None] | None,
    test_data: dict[str, np.ndarray | None],
    lookback_len: int,
    pred_len: int,
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None] | None,
    dict[str, np.ndarray | None],
]:
    """Convert each split into rolling windows for input and prediction targets.

    Args:
        train_data: Training split dictionary.
        val_data: Optional validation split dictionary.
        test_data: Test split dictionary.
        lookback_len: Input sequence length.
        pred_len: Prediction sequence length.

    Returns:
        Truncated train, optional val, and test datasets.

    """

    train_set = _window_split(train_data, lookback_len, pred_len)
    val_set = _window_split(val_data, lookback_len, pred_len) if val_data is not None else None
    test_set = _window_split(test_data, lookback_len, pred_len)

    assert len(train_set["X_N"]) > 0, (
        "Lookback length is too long for the training data."
    )
    assert len(test_set["X_N"]) > 0, "Lookback length is too long for the test data."

    for name, dset in [("train", train_set), ("test", test_set)]:
        n_windows = len(dset["X_N"])
        assert len(dset["Y_N"]) == n_windows, \
            f"{name}: X/Y window count mismatch"
        for key in ["X_N", "X_NA", "X_N_mask", "X_NA_mask", "X_A_mask"]:
            assert dset[key].shape[0] == n_windows, \
                f"{name}: {key} has wrong window count"
            assert dset[key].shape[1] == lookback_len, \
                f"{name}: {key} has wrong lookback_len"
        for key in ["Y_N", "Y_N_mask"]:
            assert dset[key].shape[0] == n_windows, \
                f"{name}: {key} has wrong window count"
            assert dset[key].shape[1] == pred_len, \
                f"{name}: {key} has wrong pred_len"
        if dset["X"] is not None:
            assert dset["X"].shape[:2] == (n_windows, lookback_len), \
                f"{name}: X has wrong shape"
        if dset["Y"] is not None:
            assert dset["Y"].shape[:2] == (n_windows, pred_len), \
                f"{name}: Y has wrong shape"
        for key in ["X_N_mask", "X_NA_mask", "X_A_mask", "Y_N_mask"]:
            vals = dset[key]
            assert vals.min() >= 0.0 and vals.max() <= 1.0, \
                f"{name}: {key} values out of [0,1]"

    return train_set, val_set, test_set


def _window_truncate(values: np.ndarray, window_size: int) -> np.ndarray:
    """Slide a window over axis 0 into ``(n_windows, window_size, n_features)``."""
    windows = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=window_size,
        axis=0,
    )
    return np.ascontiguousarray(np.moveaxis(windows, -1, 1))


def _window_split(
    data: dict[str, np.ndarray | None], lookback_len: int, pred_len: int
) -> dict[str, np.ndarray | None]:
    """Convert one split dictionary into rolling windows for input and targets."""
    data_set = {}
    len_data = len(data["N_data"])
    inputs = slice(0, len_data - pred_len)
    targets = slice(lookback_len, len_data)

    for name, key in (
        ("X_N", "N_data"),
        ("X_N_mask", "N_mask"),
        ("X_NA", "NA_data"),
        ("X_NA_mask", "NA_mask"),
        ("X_A_mask", "A_mask"),
        ("X", "data"),
    ):
        values = data[key]
        data_set[name] = (
            None if values is None else _window_truncate(values[inputs, :], lookback_len)
        )
    for name, key in (("Y_N", "N_data"), ("Y_N_mask", "N_mask"), ("Y", "data")):
        values = data[key]
        data_set[name] = (
            None if values is None else _window_truncate(values[targets, :], pred_len)
        )
    return data_set


def _subset_windows(
    windows: dict[str, np.ndarray | None], indices: np.ndarray
) -> dict[str, np.ndarray | None]:
    """Select window rows by index, preserving the dictionary layout."""
    return {
        key: (None if value is None else np.ascontiguousarray(value[indices]))
        for key, value in windows.items()
    }


def _shuffled_split(
    data: np.ndarray,
    missing_rate: float,
    artificial_missing_rate: float,
    block_size: int,
    lookback_len: int,
    pred_len: int,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    logger: logging.Logger,
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None] | None,
    dict[str, np.ndarray | None],
    _IgnoreNaNScaler,
]:
    """Partition rolling windows at random instead of cutting the series by time.

    Every sliding window is one sample, so the 7:1:2 split is a random
    permutation of the window indices rather than a set of contiguous raw
    ranges.  Two consequences follow from dropping the time order, and both are
    deliberate rather than incidental:

    * The synthetic natural mask has no test span left to be confined to, so it
      is spread over the whole series and every split carries the same
      missingness regime.
    * Normalization is fitted on the raw positions the training windows cover.
      Because the windows overlap and are drawn at random, that coverage is
      nearly the whole series -- so the shuffled split also shares raw context
      between train and test far more than a chronological cut does.
    """
    if val_ratio < 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError(
            "Expected val_ratio >= 0, test_ratio > 0, "
            "and val_ratio + test_ratio < 1."
        )
    overhead = lookback_len + pred_len - 1
    if len(data) < overhead + 1:
        raise ValueError(
            f"Time series too short ({len(data)}) for lookback={lookback_len}, "
            f"pred={pred_len}."
        )

    complete, n_data, na_data, n_mask, a_mask, na_mask = generate_missing(
        data,
        missing_rate,
        artificial_missing_rate,
        block_size,
        random_seed,
        logger,
        natural_span=None,
    )

    n_windows = len(data) - overhead
    order = numpy_rng(random_seed, "split", "shuffled").permutation(n_windows)
    n_train = int(n_windows * (1 - val_ratio - test_ratio))
    n_val = int(n_windows * val_ratio) if val_ratio > 0 else 0
    # Sorted indices keep the snapshot deterministic and give the contiguous
    # loaders a stable order; the partition itself is already random.
    train_idx = np.sort(order[:n_train])
    val_idx = np.sort(order[n_train:n_train + n_val])
    test_idx = np.sort(order[n_train + n_val:])

    covered = np.zeros(len(data), dtype=bool)
    for start in train_idx:
        covered[start:start + lookback_len + pred_len] = True
    mean = np.nanmean(n_data[covered], axis=0)
    std = np.nanstd(n_data[covered], axis=0)
    std = np.where(std == 0, 1.0, std)
    scaler = _IgnoreNaNScaler(mean, std)

    full = {
        "N_data": np.nan_to_num(scaler.transform(n_data), nan=0.0),
        "N_mask": n_mask,
        "NA_data": np.nan_to_num(scaler.transform(na_data), nan=0.0),
        "NA_mask": na_mask,
        "A_mask": a_mask,
        "data": None if complete is None else scaler.transform(complete),
    }

    windows = _window_split(full, lookback_len, pred_len)
    assert len(windows["X_N"]) == n_windows, "window count changed during splitting"
    logger.info(
        "Shuffled split over the whole series: %d windows -> train %d / val %d / "
        "test %d",
        n_windows,
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return (
        _subset_windows(windows, train_idx),
        _subset_windows(windows, val_idx) if n_val else None,
        _subset_windows(windows, test_idx),
        scaler,
    )


def get_tensor_data_loader(
    dataset: dict[str, any],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_generator: torch.Generator | None = None,
) -> tuple[dict[str, DataLoader], DataLoader, bool]:
    """Build training and evaluation dataloaders from numpy datasets.

    Args:
        dataset: Dataset dictionary containing train and test arrays.
        batch_size: Batch size for all dataloaders.
        num_workers: Number of worker subprocesses for each dataloader.
        pin_memory: Whether dataloaders should allocate pinned host memory.
        train_generator: Optional RNG for training DataLoader shuffling.

    Returns:
        A tuple of train dataloaders dict, test dataloader, and
        no-ground-truth flag.

    """
    def build_train_loader(
        split: dict[str, any],
        *,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(split["X_N"]).float(),
            torch.from_numpy(split["X_NA"]).float(),
            torch.from_numpy(split["X_NA_mask"]).float(),
            torch.from_numpy(split["X_A_mask"]).float(),
            torch.from_numpy(split["Y_N"]).float(),
            torch.from_numpy(split["Y_N_mask"]).float(),
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    data_loader = {
        "train": build_train_loader(
            dataset["train"],
            shuffle=True,
            generator=train_generator,
        ),
        # Keep evaluation independent of the shuffled optimization iterator.
        # Its six-tensor layout deliberately matches validation so train/val
        # metrics use identical artificial-mask and target semantics.
        "train_eval": build_train_loader(dataset["train"], shuffle=False),
    }

    def build_eval_loader(split: dict[str, any]) -> tuple[DataLoader, bool]:
        split_no_gt = "X" not in split or split["X"] is None
        if split_no_gt:
            eval_ds = TensorDataset(
                torch.from_numpy(split["X_N"]).float(),
                torch.from_numpy(split["X_N_mask"]).float(),
                torch.from_numpy(split["Y_N"]).float(),
                torch.from_numpy(split["Y_N_mask"]).float(),
            )
        else:
            eval_ds = TensorDataset(
                torch.from_numpy(split["X_N"]).float(),
                torch.from_numpy(split["X_N_mask"]).float(),
                torch.from_numpy(split["X"]).float(),
                torch.from_numpy(split["Y"]).float(),
            )
        loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        loader.no_gt = split_no_gt
        return loader, split_no_gt

    if "val" in dataset:
        # The validation loader must keep artificial masks and the
        # corresponding observed ground truth, even when the public eval split
        # has no complete natural-missing ground truth.
        data_loader["val"] = build_train_loader(dataset["val"], shuffle=False)

    eval_loader, no_gt = build_eval_loader(dataset["test"])

    return data_loader, eval_loader, no_gt
