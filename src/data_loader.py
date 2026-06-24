"""Data preprocessing and dataloader construction utilities."""

import hashlib
import json
import logging
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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
        val_ratio: Validation split ratio.
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
        value: float | Path | dict | list | tuple,
    ) -> float | str | dict | list:
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
    dataset_path = Path(save_folder) / "dataset.h5"
    scaler_path = Path(save_folder) / "scaler.pkl"
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
    rng = np.random.default_rng(random_seed)

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
            rng.random((int(np.ceil(len(data) / block_size)), data.shape[1]))
            < missing_rate
        )
        natural_mask = np.repeat(natural_mask_block, block_size, axis=0)[: len(data)]
        natural_missing_data = data.copy()
        natural_missing_data[natural_mask] = np.nan

    observed_len = np.sum(~natural_mask)
    artificial_missing_flags = rng.random(observed_len) < artificial_missing_rate
    artificial_mask = np.zeros(data.shape, dtype=bool)
    artificial_mask[~natural_mask] = artificial_missing_flags

    missing_mask = natural_mask | artificial_mask
    incomplete_data = data.copy()
    incomplete_data[missing_mask] = np.nan

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
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None],
]:
    """Split raw time series into contiguous, non-overlapping blocks.

    Each block is windowed independently by ``truncate_data`` later in the
    pipeline.  Splits are cut at the raw time-step level with compensation
    for the fixed windowing overhead, so the window-count ratio matches the
    configured ``val_ratio`` / ``test_ratio`` exactly.

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
        test_ratio: Test split ratio (applied to window counts).

    Returns:
        Tuple of train, validation, and test dictionaries.

    """
    total_len = len(N_data)
    overhead = lookback_len + pred_len - 1  # fixed cost per split for windowing
    min_len = overhead + 1  # at least 1 window per split
    if total_len < 3 * min_len:
        raise ValueError(
            f"Time series too short ({total_len}) to create non-overlapping "
            f"splits with lookback={lookback_len}, pred={pred_len}."
        )

    # Total windows available across non-overlapping raw blocks.
    total_windows = total_len - 3 * overhead

    train_windows = int(total_windows * (1 - val_ratio - test_ratio))
    val_windows = int(total_windows * val_ratio)
    test_windows = total_windows - train_windows - val_windows

    train_raw_len = train_windows + overhead
    val_raw_len = val_windows + overhead
    test_raw_len = total_len - train_raw_len - val_raw_len  # remainder → test

    train_start = 0
    train_end = train_raw_len

    val_start = train_end
    val_end = val_start + val_raw_len

    test_start = val_end
    test_end = total_len

    def _make_split(start: int, end: int) -> dict[str, np.ndarray | None]:
        return {
            "N_data": N_data[start:end, :],
            "N_mask": N_mask[start:end, :],
            "NA_data": NA_data[start:end, :],
            "NA_mask": NA_mask[start:end, :],
            "A_mask": A_mask[start:end, :],
            "data": data[start:end, :] if data is not None else None,
        }

    return _make_split(train_start, train_end), _make_split(val_start, val_end), _make_split(test_start, test_end)


def standardize_data(
    train_data: dict[str, np.ndarray | None],
    val_data: dict[str, np.ndarray | None],
    test_data: dict[str, np.ndarray | None],
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None],
    StandardScaler,
]:
    """Fit a scaler on train data and transform all data partitions.

    Args:
        train_data: Training split dictionary.
        val_data: Validation split dictionary.
        test_data: Test split dictionary.

    Returns:
        Scaled train/validation/test dictionaries and the fitted scaler.

    """
    scaler = StandardScaler()

    scaler.fit(train_data["N_data"])

    train_data_scaled = train_data.copy()
    val_data_scaled = val_data.copy()
    test_data_scaled = test_data.copy()

    train_data_scaled["N_data"] = scaler.transform(train_data["N_data"])
    val_data_scaled["N_data"] = scaler.transform(val_data["N_data"])
    test_data_scaled["N_data"] = scaler.transform(test_data["N_data"])
    train_data_scaled["NA_data"] = scaler.transform(train_data["NA_data"])
    val_data_scaled["NA_data"] = scaler.transform(val_data["NA_data"])
    test_data_scaled["NA_data"] = scaler.transform(test_data["NA_data"])
    train_data_scaled["data"] = (
        scaler.transform(train_data["data"]) if train_data["data"] is not None else None
    )
    val_data_scaled["data"] = (
        scaler.transform(val_data["data"]) if val_data["data"] is not None else None
    )
    test_data_scaled["data"] = (
        scaler.transform(test_data["data"]) if test_data["data"] is not None else None
    )

    return train_data_scaled, val_data_scaled, test_data_scaled, scaler


def white_nan_with_zero(
    train_data: dict[str, np.ndarray | None],
    val_data: dict[str, np.ndarray | None],
    test_data: dict[str, np.ndarray | None],
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None],
]:
    """Replace NaN values in model inputs with zeros.

    Args:
        train_data: Training split dictionary.
        val_data: Validation split dictionary.
        test_data: Test split dictionary.

    Returns:
        Train, validation, and test dictionaries with NaNs replaced by zeros.

    """
    train_data_filled = train_data.copy()
    val_data_filled = val_data.copy()
    test_data_filled = test_data.copy()

    train_data_filled["N_data"] = np.nan_to_num(train_data["N_data"], nan=0.0)
    val_data_filled["N_data"] = np.nan_to_num(val_data["N_data"], nan=0.0)
    test_data_filled["N_data"] = np.nan_to_num(test_data["N_data"], nan=0.0)
    train_data_filled["NA_data"] = np.nan_to_num(train_data["NA_data"], nan=0.0)
    val_data_filled["NA_data"] = np.nan_to_num(val_data["NA_data"], nan=0.0)
    test_data_filled["NA_data"] = np.nan_to_num(test_data["NA_data"], nan=0.0)

    return train_data_filled, val_data_filled, test_data_filled


def truncate_data(
    train_data: dict[str, np.ndarray | None],
    val_data: dict[str, np.ndarray | None],
    test_data: dict[str, np.ndarray | None],
    lookback_len: int,
    pred_len: int,
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None],
    dict[str, np.ndarray | None],
]:
    """Convert each split into rolling windows for input and prediction targets.

    Args:
        train_data: Training split dictionary.
        val_data: Validation split dictionary.
        test_data: Test split dictionary.
        lookback_len: Input sequence length.
        pred_len: Prediction sequence length.

    Returns:
        Truncated train, validation, and test datasets.

    """

    def truncate_data_(
        data: dict[str, np.ndarray | None], lookback_len: int, pred_len: int
    ) -> dict[str, np.ndarray | None]:
        def window_truncate(data: np.ndarray, window_size: int) -> np.ndarray:
            truncated_data = []
            num_rows = data.shape[0]
            for start in range(0, num_rows - window_size + 1):
                end = start + window_size
                truncated_data.append(data[start:end])
            return np.array(truncated_data)

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
    val_set = truncate_data_(val_data, lookback_len, pred_len)
    test_set = truncate_data_(test_data, lookback_len, pred_len)

    assert len(train_set["X_N"]) > 0, (
        "Lookback length is too long for the training data."
    )
    assert len(val_set["X_N"]) > 0, (
        "Lookback length is too long for the validation data."
    )
    assert len(test_set["X_N"]) > 0, "Lookback length is too long for the test data."

    return train_set, val_set, test_set


def save_h5(
    train_set: dict[str, np.ndarray],
    val_set: dict[str, np.ndarray],
    test_set: dict[str, np.ndarray],
    h5_path: Path,
) -> None:
    """Persist train/validation/test dictionaries into an HDF5 file.

    Args:
        train_set: Truncated training dataset dictionary.
        val_set: Truncated validation dataset dictionary.
        test_set: Truncated test dataset dictionary.
        h5_path: Output HDF5 file path.

    Returns:
        None.

    """
    with h5py.File(h5_path, "w") as h5f:
        train_grp = h5f.create_group("train")
        for key, value in train_set.items():
            if value is not None:
                train_grp.create_dataset(key, data=value)

        val_grp = h5f.create_group("val")
        for key, value in val_set.items():
            if value is not None:
                val_grp.create_dataset(key, data=value)

        test_grp = h5f.create_group("test")
        for key, value in test_set.items():
            if value is not None:
                test_grp.create_dataset(key, data=value)


def load_h5(h5_path: Path) -> dict[str, any]:
    """Load dataset splits from an HDF5 file.

    Args:
        h5_path: Input HDF5 file path.

    Returns:
        Dictionary with train, val, and test split mappings.

    """
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    data = {}

    with h5py.File(h5_path, "r") as h5f:
        train_set = {key: h5f["train"][key][()] for key in h5f["train"]}

        val_set = {key: h5f["val"][key][()] for key in h5f["val"]}

        test_set = {key: h5f["test"][key][()] for key in h5f["test"]}

        data = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }

    return data


def get_tensor_data_loader(
    dataset: dict[str, any],
    batch_size: int,
) -> tuple[dict[str, DataLoader], dict[str, DataLoader], bool]:
    """Build training and evaluation dataloaders from numpy datasets.

    Args:
        dataset: Dataset dictionary containing train, val, and test arrays.
        batch_size: Batch size for all dataloaders.

    Returns:
        A tuple of train/val dataloaders, test dataloader, and no-ground-truth flag.

    """
    train_ds = TensorDataset(
        torch.from_numpy(dataset["train"]["X_N"]).float(),
        torch.from_numpy(dataset["train"]["X_NA"]).float(),
        torch.from_numpy(dataset["train"]["X_NA_mask"]).float(),
        torch.from_numpy(dataset["train"]["X_A_mask"]).float(),
        torch.from_numpy(dataset["train"]["Y_N"]).float(),
        torch.from_numpy(dataset["train"]["Y_N_mask"]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(dataset["val"]["X_N"]).float(),
        torch.from_numpy(dataset["val"]["X_NA"]).float(),
        torch.from_numpy(dataset["val"]["X_NA_mask"]).float(),
        torch.from_numpy(dataset["val"]["X_A_mask"]).float(),
        torch.from_numpy(dataset["val"]["Y_N"]).float(),
        torch.from_numpy(dataset["val"]["Y_N_mask"]).float(),
    )
    data_loader = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    }

    no_gt = "X" not in dataset["test"] or dataset["test"]["X"] is None
    if no_gt:
        test_eval_ds = TensorDataset(
            torch.from_numpy(dataset["test"]["X_N"]).float(),
            torch.from_numpy(dataset["test"]["X_N_mask"]).float(),
            torch.from_numpy(dataset["test"]["Y_N"]).float(),
            torch.from_numpy(dataset["test"]["Y_N_mask"]).float(),
        )
    else:
        test_eval_ds = TensorDataset(
            torch.from_numpy(dataset["test"]["X_N"]).float(),
            torch.from_numpy(dataset["test"]["X_N_mask"]).float(),
            torch.from_numpy(dataset["test"]["X"]).float(),
            torch.from_numpy(dataset["test"]["Y"]).float(),
        )
    eval_loader = DataLoader(test_eval_ds, batch_size=batch_size, shuffle=False)

    return data_loader, eval_loader, no_gt
