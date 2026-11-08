# tests/test_model.py
import torch

from src.model import build_model


def test_output_shape():
    # 2 images in, 4 class scores per image out
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 256, 352))
    assert output.shape == (2, 4)


def test_final_layer_is_the_new_four_class_layer():
    model = build_model(pretrained=False)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 4


def test_custom_number_of_classes():
    model = build_model(num_classes=3, pretrained=False)
    assert model.fc.out_features == 3


def test_parameter_count_is_about_11_million():
    model = build_model(pretrained=False)
    total = 0
    for p in model.parameters():
        total = total + p.numel()
    assert 11_000_000 < total < 11_400_000


def test_all_parameters_are_trainable():
    # We fine-tune every layer, nothing is frozen
    model = build_model(pretrained=False)
    for p in model.parameters():
        assert p.requires_grad is True


def test_output_is_finite():
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 256, 352))
    assert torch.isfinite(output).all()


def test_softmax_sums_to_one():
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.randn(3, 3, 256, 352)), dim=1)
    for i in range(3):
        assert abs(probs[i].sum().item() - 1.0) < 1e-5


def test_same_seed_gives_the_same_model():
    x = torch.randn(1, 3, 64, 88)
    torch.manual_seed(0)
    model_a = build_model(pretrained=False).eval()
    torch.manual_seed(0)
    model_b = build_model(pretrained=False).eval()
    with torch.no_grad():
        assert torch.allclose(model_a(x), model_b(x))


def test_eval_mode_is_deterministic():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 64, 88)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    assert torch.equal(first, second)
