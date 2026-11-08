# tests/test_data.py
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FOLDS_CSV = ROOT / "data" / "folds.csv"
CLASS_NAMES = ["Abnormal", "History_of_MI", "MI", "Normal"]

# Skip this whole file when the CSV is missing (for example on a fresh clone)
pytestmark = pytest.mark.skipif(
    not FOLDS_CSV.exists(), reason="data/folds.csv not found"
)


def load_folds():
    return pd.read_csv(FOLDS_CSV)


def test_columns_exist():
    df = load_folds()
    for column in ["path", "label", "hash", "fold"]:
        assert column in df.columns


def test_known_dataset_size():
    # 928 raw files were reduced to 491 unique images by hash deduplication
    df = load_folds()
    assert len(df) == 491


def test_labels_are_the_four_classes():
    df = load_folds()
    assert set(df["label"].unique()) == set(CLASS_NAMES)


def test_class_counts():
    df = load_folds()
    counts = df["label"].value_counts().to_dict()
    assert counts["Abnormal"] == 233
    assert counts["Normal"] == 142
    assert counts["History_of_MI"] == 86
    assert counts["MI"] == 30


def test_no_duplicate_hashes():
    # A duplicate image in the index could land in both train and validation
    df = load_folds()
    assert df["hash"].is_unique


def test_folds_are_zero_to_four():
    df = load_folds()
    assert sorted(df["fold"].unique().tolist()) == [0, 1, 2, 3, 4]


def test_every_fold_has_every_class():
    df = load_folds()
    for fold in range(5):
        labels_in_fold = set(df[df["fold"] == fold]["label"])
        assert labels_in_fold == set(CLASS_NAMES)


def test_folds_have_similar_size():
    df = load_folds()
    sizes = df["fold"].value_counts().tolist()
    assert max(sizes) - min(sizes) <= 2


def test_mi_is_spread_evenly():
    # Stratified split: 30 MI images -> 6 in every fold
    df = load_folds()
    mi = df[df["label"] == "MI"]
    for fold in range(5):
        assert (mi["fold"] == fold).sum() == 6


def test_image_files_exist():
    # Check the first 20 image paths (skipped when the images are not on this machine)
    df = load_folds()
    first_path = ROOT / df.iloc[0]["path"]
    if not first_path.exists():
        pytest.skip("image files are not on this machine")
    for i in range(20):
        assert (ROOT / df.iloc[i]["path"]).exists()
