import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction="mean"):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs and targets are the same size
        if inputs.size() != targets.size():
            raise ValueError("Size of inputs and targets must be the same")

        # Sigmoid activation to get probabilities for each class
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Compute the focal loss components
        focal_weight = (1 - pt) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        if isinstance(self.alpha, torch.Tensor) and focal_weight.is_cuda:
            self.alpha = self.alpha.to("cuda")

        # Combine components
        focal_loss = self.alpha * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Sigmoid activation to get the probability predictions
        probas = torch.sigmoid(inputs)

        # Compute the pt term: pt is sigmoid_p if target is 1, and (1 - sigmoid_p) if target is 0
        pt = torch.where(targets == 1, probas, 1 - probas)

        focal_weight = (1 - pt) ** self.gamma
        bce_loss = torch.log(pt)

        if isinstance(self.alpha, torch.Tensor) and focal_weight.is_cuda:
            self.alpha = self.alpha.to("cuda")

        # Compute the focal loss
        focal_loss = -self.alpha * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
