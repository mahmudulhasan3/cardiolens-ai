# tests/test_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_model
from src.train import evaluate, train_one_epoch


def make_fake_data(n=16):
    """Small random images with labels 0,1,2,3,0,1,... (small size keeps the test fast)."""
    torch.manual_seed(0)
    images = torch.randn(n, 3, 64, 88)
    labels = torch.tensor([i % 4 for i in range(n)])
    return images, labels


def make_loader(n=16, batch_size=8):
    images, labels = make_fake_data(n)
    return (
        DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False),
        labels,
    )


def test_model_can_overfit_one_batch():
    # Sanity check: if the loss cannot go down on 8 images, something is broken
    images, labels = make_fake_data(n=8)
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    losses = []
    for step in range(30):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5


def test_train_one_epoch_updates_weights():
    loader, _ = make_loader()
    model = build_model(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = model.fc.weight.detach().clone()
    loss = train_one_epoch(model, loader, criterion, optimizer, "cpu")
    after = model.fc.weight.detach().clone()

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.equal(before, after)


def test_evaluate_does_not_change_weights():
    # Validation must only measure, it must never learn
    loader, _ = make_loader()
    model = build_model(pretrained=False)
    criterion = nn.CrossEntropyLoss()

    before = model.fc.weight.detach().clone()
    evaluate(model, loader, criterion, "cpu")
    after = model.fc.weight.detach().clone()

    assert torch.equal(before, after)
    assert model.training is False


def test_evaluate_returns_the_expected_values():
    loader, labels = make_loader()
    model = build_model(pretrained=False)
    criterion = nn.CrossEntropyLoss()

    avg_loss, macro_f1, recalls, all_labels, all_preds = evaluate(
        model, loader, criterion, "cpu"
    )

    assert avg_loss > 0
    assert 0.0 <= macro_f1 <= 1.0
    assert len(recalls) == 4 #type: ignore
    for r in recalls:  # type: ignore
        assert 0.0 <= r <= 1.0
    assert all_labels == labels.tolist()
    assert len(all_preds) == 16
    for p in all_preds:
        assert p in [0, 1, 2, 3]


def test_class_weights_make_a_missed_rare_class_cost_more():
    # Three correct predictions and one confident mistake on class 2 (like MI)
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 3, 2])

    plain = nn.CrossEntropyLoss()(logits, labels).item()
    weights = torch.tensor([1.0, 1.0, 4.0, 1.0])
    weighted = nn.CrossEntropyLoss(weight=weights)(logits, labels).item()

    assert weighted > plain


def run_one_epoch_with_seed(seed):
    loader, _ = make_loader()
    torch.manual_seed(seed)
    model = build_model(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_one_epoch(model, loader, criterion, optimizer, "cpu")


def test_same_seed_gives_the_same_training_loss():
    first = run_one_epoch_with_seed(42)
    second = run_one_epoch_with_seed(42)
    assert abs(first - second) < 1e-4
