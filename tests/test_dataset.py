# tests/test_dataset.py
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IMG_SIZE,
    ECGDataset,
    get_eval_transform,
)


def make_tiny_dataset(tmp_path):
    """Create 8 small fake images and a folds.csv for them."""
    rows = []
    for i in range(8):
        label = CLASS_NAMES[i % 4]
        fold = i % 2
        image_name = "img_" + str(i) + ".png"
        Image.new("RGB", (120, 90), color=(i * 20, 100, 200)).save(
            tmp_path / image_name
        )
        rows.append(
            {"path": image_name, "label": label, "hash": "hash" + str(i), "fold": fold}
        )
    csv_path = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_class_order_is_fixed():
    # The model output index is tied to this order, it must never change
    assert CLASS_NAMES == ["Abnormal", "History_of_MI", "MI", "Normal"]
    assert CLASS_TO_IDX["MI"] == 2


def test_image_size_constant():
    assert IMG_SIZE == (256, 352)


def test_dataset_length_follows_folds(tmp_path):
    csv_path = make_tiny_dataset(tmp_path)
    assert len(ECGDataset(csv_path, tmp_path, [0])) == 4
    assert len(ECGDataset(csv_path, tmp_path, [1])) == 4
    assert len(ECGDataset(csv_path, tmp_path, [0, 1])) == 8


def test_only_requested_folds_are_loaded(tmp_path):
    csv_path = make_tiny_dataset(tmp_path)
    dataset = ECGDataset(csv_path, tmp_path, [0])
    for i in range(len(dataset)):
        assert dataset.df.iloc[i]["fold"] == 0


def test_item_has_the_right_shape_and_type(tmp_path):
    csv_path = make_tiny_dataset(tmp_path)
    dataset = ECGDataset(csv_path, tmp_path, [0], transform=get_eval_transform())
    image, label = dataset[0]
    assert image.shape == (3, 256, 352)  # type: ignore
    assert image.dtype == torch.float32  # type: ignore
    assert isinstance(label, int)
    assert 0 <= label < 4


def test_label_matches_the_class_name(tmp_path):
    csv_path = make_tiny_dataset(tmp_path)
    dataset = ECGDataset(csv_path, tmp_path, [0, 1], transform=get_eval_transform())
    for i in range(len(dataset)):
        _, label = dataset[i]
        expected = CLASS_TO_IDX[dataset.df.iloc[i]["label"]]
        assert label == expected


def test_grayscale_image_becomes_three_channels(tmp_path):
    # The model expects RGB, a grayscale file must still work
    Image.new("L", (120, 90), color=128).save(tmp_path / "gray.png")
    row = {"path": "gray.png", "label": "Normal", "hash": "h", "fold": 0}
    csv_path = tmp_path / "folds.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    dataset = ECGDataset(csv_path, tmp_path, [0], transform=get_eval_transform())
    image, _ = dataset[0]
    assert image.shape[0] == 3  # type: ignore


def test_normalization_of_a_white_image():
    # White pixel = 1.0, after normalization it must be (1 - mean) / std per channel
    white = Image.new("RGB", (100, 100), color=(255, 255, 255))
    tensor = get_eval_transform()(white)
    expected = [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
    for channel in range(3):
        assert (
            abs(tensor[channel].mean().item() - expected[channel]) < 1e-3  # type: ignore
        )


def test_dataloader_makes_batches(tmp_path):
    csv_path = make_tiny_dataset(tmp_path)
    dataset = ECGDataset(csv_path, tmp_path, [0, 1], transform=get_eval_transform())
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 256, 352)
    assert labels.shape == (4,)
