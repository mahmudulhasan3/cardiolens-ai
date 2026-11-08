# tests/test_predict.py
import json
from pathlib import Path

import pytest
from PIL import Image

from src.dataset import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "resnet18_final.pt"
META = ROOT / "models" / "resnet18_final.json"

# These tests need the trained final model, skip them when it is not on this machine
pytestmark = pytest.mark.skipif(
    not (WEIGHTS.exists() and META.exists()), reason="final model files not found"
)


@pytest.fixture(scope="module")
def predictor():
    from src.predict import ECGPredictor

    return ECGPredictor()


def fake_ecg():
    return Image.new("RGB", (900, 600), color=(255, 255, 255))


def test_metadata_file():
    meta = json.loads(META.read_text())
    assert meta["class_names"] == CLASS_NAMES
    assert meta["img_size"] == [256, 352]
    assert "cv_macro_f1_mean" in meta


def test_model_is_in_eval_mode(predictor):
    assert predictor.model.training is False


def test_result_has_the_expected_keys(predictor):
    result = predictor.predict(fake_ecg())
    assert set(result.keys()) == {"predicted_class", "confidence", "probabilities"}
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)


def test_probabilities_sum_to_one(predictor):
    result = predictor.predict(fake_ecg())
    total = 0
    for name in result["probabilities"]:
        total = total + result["probabilities"][name]
    assert abs(total - 1.0) < 1e-3


def test_prediction_is_the_most_likely_class(predictor):
    result = predictor.predict(fake_ecg())
    best_name = max(result["probabilities"], key=result["probabilities"].get)
    assert result["predicted_class"] == best_name
    assert result["confidence"] == result["probabilities"][best_name]
    assert 0.0 <= result["confidence"] <= 1.0


def test_same_image_gives_the_same_result(predictor):
    first = predictor.predict(fake_ecg())
    second = predictor.predict(fake_ecg())
    assert first == second


def test_predict_accepts_a_file_path(predictor, tmp_path):
    path = tmp_path / "ecg.png"
    fake_ecg().save(path)
    result = predictor.predict(str(path))
    assert result["predicted_class"] in CLASS_NAMES


def test_grayscale_and_rgba_images_work(predictor):
    gray = Image.new("L", (900, 600), color=200)
    rgba = Image.new("RGBA", (900, 600), color=(255, 255, 255, 255))
    assert predictor.predict(gray)["predicted_class"] in CLASS_NAMES
    assert predictor.predict(rgba)["predicted_class"] in CLASS_NAMES
