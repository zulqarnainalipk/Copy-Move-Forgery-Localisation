"""Segmentation losses on logits; targets are float masks in {0, 1}."""
import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    """Soft Dice loss averaged over the batch."""
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probabilities * targets).sum(1)
    return (1 - (2 * intersection + eps) / (probabilities.sum(1) + targets.sum(1) + eps)).mean()


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Binary focal loss with class weight alpha and focusing parameter gamma."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.exp(-bce)
    weight = alpha * targets + (1 - alpha) * (1 - targets)
    return (weight * (1 - probability) ** gamma * bce).mean()


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1e-6):
    """Tversky loss weighting false positives by alpha and false negatives by beta."""
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    tp = (probabilities * targets).sum(1)
    fp = (probabilities * (1 - targets)).sum(1)
    fn = ((1 - probabilities) * targets).sum(1)
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def build_loss(name):
    """Return the loss function for a name: bce, bce_dice, focal or tversky."""
    bce = F.binary_cross_entropy_with_logits
    table = {
        "bce": lambda x, y: bce(x, y),
        "bce_dice": lambda x, y: bce(x, y) + dice_loss(x, y),
        "focal": focal_loss,
        "tversky": lambda x, y: bce(x, y) + tversky_loss(x, y),
    }
    return table[name]
