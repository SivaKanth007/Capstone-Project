"""
Dataset Download Module
========================
Downloads the NASA C-MAPSS turbofan engine degradation dataset from Kaggle.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def ensure_kaggle_installed():
    """Ensure kaggle CLI is available."""
    try:
        import kaggle
        print("[DOWNLOAD] Kaggle package found.")
    except ImportError:
        print("[DOWNLOAD] Installing kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])


def download_cmapss(dataset_id=None, output_dir=None):
    """
    Download NASA C-MAPSS dataset from Kaggle.

    Parameters
    ----------
    dataset_id : str
        Kaggle dataset identifier (default from config).
    output_dir : str
        Directory to save downloaded files.
    """
    dataset_id = dataset_id or config.CMAPSS_DATASET
    output_dir = output_dir or config.RAW_DATA_DIR

    ensure_kaggle_installed()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"[DOWNLOAD] Downloading {dataset_id} to {output_dir}...")
    api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
    print(f"[DOWNLOAD] Download complete. Files:")
    for f in os.listdir(output_dir):
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
        print(f"  - {f} ({size_mb:.2f} MB)")


def load_cmapss_train(subset="FD001", data_dir=None):
    """
    Load C-MAPSS training data.

    Returns
    -------
    pd.DataFrame
        Training data with named columns and computed RUL.
    """
    data_dir = data_dir or config.RAW_DATA_DIR

    # Try various common file naming patterns
    possible_names = [
        f"train_{subset}.txt",
        f"train_{subset}.csv",
        f"PM_train.txt",
    ]

    filepath = None
    for name in possible_names:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            filepath = candidate
            break

    # Search recursively if not found at top level
    if filepath is None:
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if "train" in f.lower() and subset.lower() in f.lower():
                    filepath = os.path.join(root, f)
                    break
                elif "pm_train" in f.lower():
                    filepath = os.path.join(root, f)
                    break
            if filepath:
                break

    if filepath is None:
        raise FileNotFoundError(
            f"Could not find training file for {subset} in {data_dir}. "
            f"Available files: {os.listdir(data_dir)}"
        )

    print(f"[DOWNLOAD] Loading training data from: {filepath}")

    df = pd.read_csv(filepath, sep=r"\s+", header=None)

    # Assign column names
    if len(df.columns) == len(config.CMAPSS_COLUMNS):
        df.columns = config.CMAPSS_COLUMNS
    else:
        # Handle extra columns sometimes present
        cols = config.CMAPSS_COLUMNS[:len(df.columns)]
        df.columns = cols

    # Compute RUL (Remaining Useful Life) for each unit
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    # Cap RUL at MAX_RUL (piecewise linear degradation)
    df["RUL"] = df["RUL"].clip(upper=config.MAX_RUL)

    print(f"[DOWNLOAD] Loaded {len(df)} rows, {df['unit_id'].nunique()} units")
    print(f"[DOWNLOAD] Cycle range: {df['cycle'].min()} - {df['cycle'].max()}")
    print(f"[DOWNLOAD] RUL range: {df['RUL'].min()} - {df['RUL'].max()}")

    return df


def load_cmapss_test(subset="FD001", data_dir=None):
    """
    Load C-MAPSS test data and RUL labels.

    Returns
    -------
    tuple(pd.DataFrame, pd.Series)
        Test data and true RUL values.
    """
    data_dir = data_dir or config.RAW_DATA_DIR

    # Find test file
    test_path = None
    rul_path = None

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            fl = f.lower()
            if ("test" in fl and subset.lower() in fl) or "pm_test" in fl:
                test_path = test_path or os.path.join(root, f)
            if ("rul" in fl and subset.lower() in fl) or "pm_truth" in fl:
                rul_path = rul_path or os.path.join(root, f)

    if test_path is None:
        raise FileNotFoundError(f"Could not find test file for {subset} in {data_dir}")

    print(f"[DOWNLOAD] Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path, sep=r"\s+", header=None)

    if len(df_test.columns) == len(config.CMAPSS_COLUMNS):
        df_test.columns = config.CMAPSS_COLUMNS
    else:
        df_test.columns = config.CMAPSS_COLUMNS[:len(df_test.columns)]

    # Load RUL truth
    rul_true = None
    if rul_path:
        print(f"[DOWNLOAD] Loading RUL labels from: {rul_path}")
        rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None)
        rul_true = rul_true.iloc[:, 0]
        rul_true.name = "RUL_true"

    return df_test, rul_true


if __name__ == "__main__":
    print("=" * 60)
    print("NASA C-MAPSS Dataset Download & Loading")
    print("=" * 60)

    # Step 1: Download
    download_cmapss()

    # Step 2: Load and verify
    df_train = load_cmapss_train()
    print(f"\nTraining data shape: {df_train.shape}")
    print(df_train.head())

    df_test, rul_true = load_cmapss_test()
    print(f"\nTest data shape: {df_test.shape}")
    if rul_true is not None:
        print(f"RUL labels: {len(rul_true)} entries")
