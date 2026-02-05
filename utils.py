import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py
# https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
# https://arxiv.org/pdf/1708.02002.pdf        

class SoftmaxFocalLoss(nn.Module):
    """
    Multi-class focal loss (softmax-based).
    logits: [B, C]
    target: [B] (int class indices)
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        target = target.long()  # [B]
        log_probs = F.log_softmax(logits, dim=1)  # [B, C]
        probs = torch.exp(log_probs)              # [B, C]

        # CE per-sample (no reduction)
        
        ce = F.nll_loss(
            log_probs, target,
            weight=self.weight,
            reduction="none"
        )  # [B]


        # pt = prob of the true class
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]

        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()