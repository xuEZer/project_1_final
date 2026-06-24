"""Collect variate-level metadata for all TIME benchmark datasets into SQLite.

Each row = one variate (1-D time series) within one item of one dataset-frequency.
Columns: dataset, frequency, item_id, variate_name, variate_index, length, missing_rate.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from huggingface_hub import list_repo_tree, snapshot_download

DB_PATH = Path(__file__).resolve().parent / "time_metadata.sqlite"
REPO_ID = "Real-TSF/TIME"


def discover_datasets() -> list[tuple[str, list[str]]]:
    """Return [(dataset_name, [frequency, ...]), ...] for all TIME datasets."""
    tree = list(list_repo_tree(REPO_ID, repo_type="dataset", recursive=False))
    result: list[tuple[str, list[str]]] = []
    for item in tree:
        if item.path in (".gitattributes", "README.md"):
            continue
        ds_name = item.path
        sub = list(
            list_repo_tree(REPO_ID, repo_type="dataset", recursive=False,
                           path_in_repo=ds_name)
        )
        freqs = sorted(
            s.path.split("/")[-1]
            for s in sub
            if not s.path.startswith(".")
        )
        result.append((ds_name, freqs))
    return result


def process_dataset(db: sqlite3.Connection, ds_name: str, freq: str) -> None:
    """Download, load, and insert variate-level metadata for one dataset-frequency."""
    print(f"  {ds_name}/{freq} ...", end=" ", flush=True)
    try:
        snap = snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision="main",
            allow_patterns=f"{ds_name}/{freq}/*",
        )
        ds = load_from_disk(f"{snap}/{ds_name}/{freq}")
    except Exception as e:
        print(f"SKIP (download/load error: {e})")
        return

    rows: list[tuple] = []
    for item_idx in range(len(ds)):
        item = ds[item_idx]
        item_id = item["item_id"]
        target = np.array(item["target"])  # (num_variates, time_steps)
        variate_names = item.get("variate_names", None)

        if target.ndim == 1:
            target = target.reshape(1, -1)

        num_variates, length = target.shape
        for vi in range(num_variates):
            col = target[vi]
            missing_rate = float(np.isnan(col).mean())
            vname = variate_names[vi] if variate_names and vi < len(variate_names) else str(vi)
            rows.append((ds_name, freq, item_id, vname, vi, length, missing_rate))

    db.executemany(
        "INSERT OR REPLACE INTO variates VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.commit()
    print(f"{len(ds)} items × {len(rows)//max(1,len(ds))} variates = {len(rows)} variates")


def main() -> None:
    datasets = discover_datasets()
    total_freqs = sum(len(freqs) for _, freqs in datasets)
    print(f"Found {len(datasets)} datasets, {total_freqs} frequency combinations\n")

    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS variates (
            dataset       TEXT NOT NULL,
            frequency     TEXT NOT NULL,
            item_id       TEXT NOT NULL,
            variate_name  TEXT NOT NULL,
            variate_index INTEGER NOT NULL,
            length        INTEGER NOT NULL,
            missing_rate  REAL NOT NULL,
            PRIMARY KEY (dataset, frequency, item_id, variate_index)
        )
    """)
    db.execute("DELETE FROM variates")
    db.commit()

    for ds_name, freqs in sorted(datasets):
        for freq in freqs:
            process_dataset(db, ds_name, freq)

    db.close()
    print(f"\nDone → {DB_PATH}")


if __name__ == "__main__":
    main()
