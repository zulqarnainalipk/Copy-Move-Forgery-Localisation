import pytest

torch = pytest.importorskip("torch")

from forgery.losses import build_loss


@pytest.mark.parametrize("name", ["bce", "bce_dice", "focal", "tversky"])
def test_losses_build_and_prefer_correct_logits(name):
    loss = build_loss(name)
    targets = torch.zeros(2, 1, 8, 8)
    targets[:, :, 2:6, 2:6] = 1
    good = loss(targets * 20 - 10, targets)
    bad = loss(10 - targets * 20, targets)
    assert torch.isfinite(good) and good.ndim == 0
    assert good < bad


def test_unknown_loss_raises():
    with pytest.raises(KeyError):
        build_loss("unknown")
