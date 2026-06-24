#!/usr/bin/env python3
"""Verify all dataset configs under configs/template/dataset/.

For each config, load the actual data and check whether the claimed length
and missing rate match reality.
"""

import configparser
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from huggingface_hub import snapshot_download

REPO_ID = "Real-TSF/TIME"
CONFIG_DIR = Path("configs/template/dataset")
SKIP_LARGE = False  # set True to skip datasets >50k points


def check_one(config_path: Path) -> dict:
    config = configparser.ConfigParser()
    config.read(config_path)

    ds_name = config["dataset"]["dataset_name"]
    frequency = config["dataset"]["frequency"]
    item_id = config["dataset"]["item_id"]
    col_idx = config.getint("dataset", "column_index")

    # Parse claimed values from the comment
    comment_line = config_path.read_text().splitlines()[0]
    claimed_length = None
    claimed_missing = None  # None = "complete", float = pct
    if "Length:" in comment_line:
        try:
            claimed_length = int(comment_line.split("Length:")[1].split(",")[0].strip())
        except (ValueError, IndexError):
            pass
    if "Missing rate:" in comment_line:
        try:
            claimed_missing = float(
                comment_line.split("Missing rate:")[1].split("%")[0].strip()
            )
        except (ValueError, IndexError):
            pass
    elif "Complete" in comment_line:
        claimed_missing = None  # means 0%

    # Download & load
    try:
        snap = snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision="main",
            allow_patterns=f"{ds_name}/{frequency}/*",
        )
        ds = load_from_disk(f"{snap}/{ds_name}/{frequency}")
    except Exception as e:
        return {
            "file": config_path.name,
            "error": f"download/load: {e}",
        }

    # Find the item
    filtered = ds.filter(lambda x: x["item_id"] == item_id)
    if len(filtered) == 0:
        return {
            "file": config_path.name,
            "error": f"item_id '{item_id}' not found (available: {ds['item_id']})",
        }

    item = filtered[0]
    target = np.array(item["target"])
    if target.ndim == 1:
        target = target.reshape(1, -1)

    if col_idx >= target.shape[0]:
        return {
            "file": config_path.name,
            "error": f"column_index {col_idx} out of range (max {target.shape[0]-1})",
        }

    series = target[col_idx]
    actual_length = len(series)
    actual_missing_pct = float(np.isnan(series).mean() * 100)

    # Compare
    length_ok = claimed_length is None or actual_length == claimed_length
    if claimed_missing is None:  # claimed "Complete (no missing)"
        missing_ok = actual_missing_pct == 0.0
    else:
        missing_ok = abs(actual_missing_pct - claimed_missing) < 0.05

    return {
        "file": config_path.name,
        "ds_name": ds_name,
        "frequency": frequency,
        "item_id": item_id,
        "col_idx": col_idx,
        "claimed_length": claimed_length,
        "actual_length": actual_length,
        "length_ok": length_ok,
        "claimed_missing_pct": claimed_missing if claimed_missing is not None else 0.0,
        "actual_missing_pct": round(actual_missing_pct, 2),
        "missing_ok": missing_ok,
    }


def main() -> int:
    config_files = sorted(CONFIG_DIR.glob("*.ini"))
    results = []
    errors = 0

    for cf in config_files:
        r = check_one(cf)
        results.append(r)
        if "error" in r:
            print(f"❌ {r['file']}: {r['error']}")
            errors += 1
        else:
            status = "OK" if (r["length_ok"] and r["missing_ok"]) else "MISMATCH"
            flag = "✓" if status == "OK" else "✗"
            print(
                f"{flag} {r['file']:<36} "
                f"len: claimed={r['claimed_length']:>7} actual={r['actual_length']:>7} "
                f"{'✓' if r['length_ok'] else '✗'}  "
                f"missing: claimed={r['claimed_missing_pct']:>6.2f}% actual={r['actual_missing_pct']:>6.2f}% "
                f"{'✓' if r['missing_ok'] else '✗'}"
            )
            if status != "OK":
                errors += 1

    print()
    total = len(results)
    passed = total - errors
    print(f"Total: {total}  Passed: {passed}  Failed: {errors}")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
