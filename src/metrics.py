import torch


def dice_score(preds: torch.Tensor,
               targets: torch.Tensor,
               threshold: float = 0.5,
               eps: float = 1e-7) -> float:

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (2.0 * intersection + eps) / (
        preds.sum() + targets.sum() + eps
    )

    return dice.item()


def iou_score(preds: torch.Tensor,
              targets: torch.Tensor,
              threshold: float = 0.5,
              eps: float = 1e-7) -> float:
    """
    Computes Intersection over Union (IoU).

    Parameters
    ----------
    preds : torch.Tensor
        Model logits.
    targets : torch.Tensor
        Ground truth masks.
    threshold : float
        Threshold applied after sigmoid.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        IoU score.
    """

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + eps) / (union + eps)

    return iou.item()