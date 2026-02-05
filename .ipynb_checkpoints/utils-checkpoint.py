import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py
# https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
# https://arxiv.org/pdf/1708.02002.pdf        

class SoftmaxFocalLoss(nn.Module):
    """
    Multi-class focal loss (single-label).
    - logits: [B, C]
    - targets: [B] (class indices)
    gamma=2, alpha=1 per your requirement.

    Optional: weight (Tensor[C]) for class imbalance (like CE class weights).
    """
    def __init__(self, gamma=2.0, alpha=1.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits, targets):
        # log_probs: [B, C]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()  # softmax probs

        # gather true class prob/logprob: [B]
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # focal modulating factor: [B]
        focal_factor = (1.0 - pt).pow(self.gamma)

        # base CE loss: [B]
        ce_loss = -log_pt

        # optional class weighting for imbalance
        if self.weight is not None:
            class_w = self.weight.gather(0, targets)  # [B]
            ce_loss = ce_loss * class_w

        loss = self.alpha * focal_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B]
