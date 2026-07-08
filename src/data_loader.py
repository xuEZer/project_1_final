"""Data preprocessing and dataloader construction utilities."""

import contextlib
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.rng import numpy_rng


def preprocess_dataset(
    dataset_name: str,
    frequency: str,
    item_id: str,
    column_index: int,
    missing_rate: float,
    artificial_missing_rate: float,
    block_size: int,
    lookback_len: int,
    pred_len: int,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    logger: logging.Logger,
    save_path: Path,
) -> Path:
    """Preprocess raw data and persist reusable artifacts on disk.

    Args:
        dataset_name: Dataset name on the TIME benchmark repository.
        frequency: Sampling frequency subfolder under the dataset.
        item_id: Item index within the dataset.
        column_index: Target column index for multivariate inputs.
        missing_rate: Rate used to inject natural missing blocks when needed.
        artificial_missing_rate: Rate used to inject artificial missing values.
        block_size: Block size used for natural missing simulation.
        lookback_len: Input window length.
        pred_len: Forecast horizon length.
        val_ratio: Validation split ratio (0 to disable).
        test_ratio: Test split ratio.
        random_seed: Random seed for reproducible preprocessing.
        logger: Logger used to report preprocessing progress.
        save_path: Root folder used to store cached preprocessing artifacts.

    Returns:
        Folder path containing the serialized dataset and scaler.

    """
    uid = generate_preprocess_uid(
        {
            "dataset_name": dataset_name,
            "frequency": frequency,
            "item_id": item_id,
            "column_index": column_index,
            "missing_rate": missing_rate,
            "artificial_missing_rate": artificial_missing_rate,
            "block_size": block_size,
            "lookback_len": lookback_len,
            "pred_len": pred_len,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
        }
    )
    logger.info(f"Generated preprocess UID: {uid}")

    save_folder = save_path / uid
    if check_if_data_preprocessed(save_folder):
        logger.info(f"Data already preprocessed. Loading from {save_folder}")
        return save_folder

    lock_folder = save_path / f"{uid}.lock"
    lock_acquired = False
    while not lock_acquired:
        try:
            lock_folder.mkdir(parents=True, exist_ok=False)
            lock_acquired = True
        except FileExistsError:
            logger.info(f"Waiting for preprocessing lock: {lock_folder}")
            time.sleep(5)
            if check_if_data_preprocessed(save_folder):
                logger.info(f"Data already preprocessed. Loading from {save_folder}")
                return save_folder

    try:
        if check_if_data_preprocessed(save_folder):
            logger.info(f"Data already preprocessed. Loading from {save_folder}")
            return save_folder

        save_folder.mkdir(parents=True, exist_ok=True)

        snapshot_dir = snapshot_download(
            repo_id="Real-TSF/TIME",
            repo_type="dataset",
            revision="main",
            allow_patterns=f"{dataset_name}/{frequency}/*",
        )
        dataset = load_from_disk(f"{snapshot_dir}/{dataset_name}/{frequency}")
        filtered = dataset.filter(lambda x: x["item_id"] == item_id)
        if len(filtered) == 0:
            raise ValueError(f"item_id '{item_id}' not found in dataset")
        dataset = np.array(filtered[0]["target"])
        data = dataset[column_index, :] if dataset.ndim > 1 else dataset
        data = data.reshape(-1, 1)

        data, N_data, NA_data, N_mask, A_mask, NA_mask = generate_missing(
            data, missing_rate, artificial_missing_rate, block_size, random_seed, logger
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

        save_h5(train_set, val_set, test_set, save_folder / "dataset.h5")
        joblib.dump(scaler, save_folder / "scaler.pkl")

        logger.info(f"Preprocessed data saved to {save_folder}")
        return save_folder
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_folder.rmdir()


def generate_preprocess_uid(
    preprocess_params: dict[str, float | int | str | list | dict],
) -> str:
    """Create a stable hash identifier for preprocessing parameters.

    Args:
        preprocess_params: Parameters that uniquely define preprocessing behavior.

    Returns:
        A deterministic short hash string for the provided parameters.

    """

    def _normalize_uid_value(
        value: float | int | str | Path | dict | list | tuple,
    ) -> float | int | str | dict | list:
        if isinstance(value, float):
            return format(value, ".12g")
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, dict):
            return {k: _normalize_uid_value(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [_normalize_uid_value(v) for v in value]
        return value

    normalized_params = _normalize_uid_value(preprocess_params)
    params_payload = json.dumps(
        normalized_params,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(params_payload.encode("utf-8")).hexdigest()[:16]


def check_if_data_preprocessed(save_folder: Path) -> bool:
    """Check whether preprocessing artifacts already exist in the target folder.

    Args:
        save_folder: Folder expected to contain serialized preprocessing artifacts.

    Returns:
        True if both dataset and scaler files exist, otherwise False.

    """
    dataset_path = save_folder / "dataset.h5"
    scaler_path = save_folder / "scaler.pkl"
    return dataset_path.exists() and scaler_path.exists()


def generate_missing(
    data: np.ndarray,
    missing_rate: float,
    artificial_missing_rate: float,
    block_size: int,
    random_seed: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inject natural and artificial missing values and return masks.

    Args:
        data: Input time-series array.
        missing_rate: Natural missing rate used when raw data has no NaNs.
        artificial_missing_rate: Artificial missing rate applied to observed values.
        block_size: Block size used to sample natural missing intervals.
        random_seed: Seed for random mask generation.
        logger: Logger for missing-data diagnostics.

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
        natural_mask_block = (
            natural_rng.random((int(np.ceil(len(data) / block_size)), data.shape[1]))
            < missing_rate
        )
        natural_mask = np.repeat(natural_mask_block, block_size, axis=0)[: len(data)]
        natural_missing_data = data.copy()
        natural_missing_data[natural_mask] = np.nan

    observed_len = np.sum(~natural_mask)
    artificial_missing_flags = artificial_rng.random(observed_len) < artificial_missing_rate
    artificial_mask = np.zeros(data.shape, dtype=bool)
    artificial_mask[~natural_mask] = artificial_missing_flags

    missing_mask = natural_mask | artificial_mask
    incomplete_data = data.copy()
    incomplete_data[missing_mask] = np.nan

    # --- assertions ---
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
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None] | None,
    dict[str, np.ndarray | None],
]:
    """Split raw time series into contiguous, non-overlapping blocks.

    Each block is windowed independently by ``truncate_data`` later in the
    pipeline.  Splits are cut at the raw time-step level with compensation
    for the fixed windowing overhead, so the window-count ratio approximates
    the configured ratios.  If ``val_ratio`` is 0, only train/test splits
    are created.

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

    Returns:
        Tuple of train, optional validation, and test dictionaries.

    """
    len_data = len(N_data)
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
    test_windows = total_windows - train_windows - val_windows

    train_raw_len = train_windows + overhead
    val_raw_len = val_windows + overhead if val_ratio > 0 else 0
    test_raw_len = len_data - train_raw_len - val_raw_len

    train_start = 0
    train_end = train_raw_len

    val_start = train_end
    val_end = val_start + val_raw_len

    test_start = val_end if val_ratio > 0 else train_end
    test_end = len_data

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
    val_split = _make_split(val_start, val_end) if val_ratio > 0 else None
    test_split = _make_split(test_start, test_end)

    # --- assertions ---
    _expected_keys = {"N_data", "N_mask", "NA_data", "NA_mask", "A_mask", "data"}
    assert train_split.keys() == _expected_keys
    assert test_split.keys() == _expected_keys
    assert train_split["N_data"].shape[0] > 0, "Train split must be non-empty"
    assert test_split["N_data"].shape[0] > 0, "Test split must be non-empty"
    total_split_len = train_split["N_data"].shape[0] + test_split["N_data"].shape[0]
    if val_split is not None:
        total_split_len += val_split["N_data"].shape[0]
    assert total_split_len == len_data, "Split lengths must sum to original length"

    return train_split, val_split, test_split


def standardize_data(
    train_data: dict[str, np.ndarray | None],
    val_data: dict[str, np.ndarray | None] | None,
    test_data: dict[str, np.ndarray | None],
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None] | None,
    dict[str, np.ndarray | None],
    StandardScaler,
]:
    """Fit a scaler on train data and transform all data partitions.

    Args:
        train_data: Training split dictionary.
        val_data: Optional validation split dictionary.
        test_data: Test split dictionary.

    Returns:
        Scaled train/val/test dictionaries and the fitted scaler.

    """
    scaler = StandardScaler()
    scaler.fit(train_data["N_data"])

    def _scale(split: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
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

    # --- assertions ---
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

    # --- assertions ---
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

    def truncate_data_(
        data: dict[str, np.ndarray | None], lookback_len: int, pred_len: int
    ) -> dict[str, np.ndarray | None]:
        def window_truncate(data: np.ndarray, window_size: int) -> np.ndarray:
            windows = np.lib.stride_tricks.sliding_window_view(
                data,
                window_shape=window_size,
                axis=0,
            )
            return np.ascontiguousarray(np.moveaxis(windows, -1, 1))

        data_set = {}
        len_data = len(data["N_data"])

        data_set["X_N"] = window_truncate(
            data["N_data"][: len_data - pred_len, :], lookback_len
        )
        data_set["X_N_mask"] = window_truncate(
            data["N_mask"][: len_data - pred_len, :], lookback_len
        )
        data_set["X_NA"] = window_truncate(
            data["NA_data"][: len_data - pred_len, :], lookback_len
        )
        data_set["X_NA_mask"] = window_truncate(
            data["NA_mask"][: len_data - pred_len, :], lookback_len
        )
        data_set["X_A_mask"] = window_truncate(
            data["A_mask"][: len_data - pred_len, :], lookback_len
        )
        data_set["X"] = (
            window_truncate(data["data"][: len_data - pred_len, :], lookback_len)
            if data["data"] is not None
            else None
        )

        data_set["Y_N"] = window_truncate(data["N_data"][lookback_len:, :], pred_len)
        data_set["Y_N_mask"] = window_truncate(
            data["N_mask"][lookback_len:, :], pred_len
        )
        data_set["Y"] = (
            window_truncate(data["data"][lookback_len:, :], pred_len)
            if data["data"] is not None
            else None
        )

        return data_set

    train_set = truncate_data_(train_data, lookback_len, pred_len)
    val_set = truncate_data_(val_data, lookback_len, pred_len) if val_data is not None else None
    test_set = truncate_data_(test_data, lookback_len, pred_len)

    # --- assertions ---
    assert len(train_set["X_N"]) > 0, (
        "Lookback length is too long for the training data."
    )
    assert len(test_set["X_N"]) > 0, "Lookback length is too long for the test data."

    for name, dset in [("train", train_set), ("test", test_set)]:
        n_windows = len(dset["X_N"])
        # X and Y must have the same number of windows
        assert len(dset["Y_N"]) == n_windows, \
            f"{name}: X/Y window count mismatch"
        # All X tensors must share the first dim
        for key in ["X_N", "X_NA", "X_N_mask", "X_NA_mask", "X_A_mask"]:
            assert dset[key].shape[0] == n_windows, \
                f"{name}: {key} has wrong window count"
            assert dset[key].shape[1] == lookback_len, \
                f"{name}: {key} has wrong lookback_len"
        # All Y tensors
        for key in ["Y_N", "Y_N_mask"]:
            assert dset[key].shape[0] == n_windows, \
                f"{name}: {key} has wrong window count"
            assert dset[key].shape[1] == pred_len, \
                f"{name}: {key} has wrong pred_len"
        # Optional X and Y
        if dset["X"] is not None:
            assert dset["X"].shape[:2] == (n_windows, lookback_len), \
                f"{name}: X has wrong shape"
        if dset["Y"] is not None:
            assert dset["Y"].shape[:2] == (n_windows, pred_len), \
                f"{name}: Y has wrong shape"
        # Masks must be 0/1
        for key in ["X_N_mask", "X_NA_mask", "X_A_mask", "Y_N_mask"]:
            vals = dset[key]
            assert vals.min() >= 0.0 and vals.max() <= 1.0, \
                f"{name}: {key} values out of [0,1]"

    return train_set, val_set, test_set


def save_h5(
    train_set: dict[str, np.ndarray],
    val_set: dict[str, np.ndarray] | None,
    test_set: dict[str, np.ndarray],
    h5_path: Path,
) -> None:
    """Persist train/val/test dictionaries into an HDF5 file."""
    with h5py.File(h5_path, "w") as h5f:
        for grp_name, dset in [
            ("train", train_set), ("val", val_set), ("test", test_set),
        ]:
            if dset is None:
                continue
            grp = h5f.create_group(grp_name)
            for key, value in dset.items():
                if value is not None:
                    grp.create_dataset(key, data=value)


def load_h5(h5_path: Path) -> dict[str, Any]:
    """Load dataset splits from an HDF5 file.

    Returns:
        Dictionary with ``"train"``, ``"val"`` (if present), and
        ``"test"`` split mappings.

    """
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    data: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as h5f:
        for grp_name in ("train", "val", "test"):
            if grp_name not in h5f:
                continue
            data[grp_name] = {key: h5f[grp_name][key][()] for key in h5f[grp_name]}
    return data


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
    train_ds = TensorDataset(
        torch.from_numpy(dataset["train"]["X_N"]).float(),
        torch.from_numpy(dataset["train"]["X_NA"]).float(),
        torch.from_numpy(dataset["train"]["X_NA_mask"]).float(),
        torch.from_numpy(dataset["train"]["X_A_mask"]).float(),
        torch.from_numpy(dataset["train"]["Y_N"]).float(),
        torch.from_numpy(dataset["train"]["Y_N_mask"]).float(),
    )
    data_loader = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
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
        data_loader["val"] = build_eval_loader(dataset["val"])[0]

    eval_loader, no_gt = build_eval_loader(dataset["test"])

    return data_loader, eval_loader, no_gt
