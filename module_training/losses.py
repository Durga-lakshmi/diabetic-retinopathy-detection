import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalCELoss(nn.Module):
    def __init__(
        self,
        alpha=None,              # list/tuple/tensor 或 None
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = None, # 用 alpha_0...alpha_k 时要用
        **kwargs,                # 用来接 alpha_0, alpha_1, ...
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        alpha_tensor = None

        # 1) 显式 alpha 列表
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                alpha_tensor = alpha.to(dtype=torch.float32)
            else:
                raise TypeError(f"alpha must be list/tuple/tensor or None, got {type(alpha)}")

        # 2) 从 alpha_0, alpha_1, ... 组装
        else:
            alpha_keys = [k for k in kwargs.keys() if k.startswith("alpha_")]
            if alpha_keys:
                if num_classes is None:
                    indices = [int(k.split("_")[1]) for k in alpha_keys]
                    num_classes = max(indices) + 1
                values = []
                for c in range(num_classes):
                    v = kwargs.get(f"alpha_{c}", 1.0)  # 没给的默认 1.0
                    values.append(float(v))
                alpha_tensor = torch.tensor(values, dtype=torch.float32)

        if alpha_tensor is not None:
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None  # 不加类权重

    def forward(self, logits, targets):
        # 保证 weight 和 logits 在同一块 device 上
        if self.alpha is not None:
            weight = self.alpha.to(logits.device)
        else:
            weight = None

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=weight,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets.float())
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = alpha_factor * focal_factor * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DistancePenalty(nn.Module):
    def __init__(self, weight=0.3, p=1):
        super().__init__()
        self.weight = weight
        self.p = p

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        levels = torch.arange(logits.size(1), device=logits.device)
        y_hat = (probs * levels).sum(dim=1)

        target = target.float()
        if self.p == 1:
            dist = torch.abs(y_hat - target)
        else:
            dist = (y_hat - target) ** 2

        return self.weight * dist.mean()


def tv_loss(x):
    """
    x: (B, 1, H, W) 或 (B, C, H, W)
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def feature_correlation_loss(features):
    """
    features: List[Tensor], each of shape (B, D)
    """
    loss = 0.0
    num = len(features)

    for i in range(num):
        for j in range(i + 1, num):
            cos_sim = F.cosine_similarity(features[i], features[j], dim=1)  # (B,)
            loss += cos_sim.abs().mean()
    return loss


def feature_orthogonality_loss(features):
    """
    features: List[Tensor], each of shape (B, D)
    """
    loss = 0.0
    num = len(features)

    for i in range(num):
        fi = F.normalize(features[i], dim=1)
        for j in range(i + 1, num):
            fj = F.normalize(features[j], dim=1)
            cos_sim = (fi * fj).sum(dim=1)   # (B,)
            loss += (cos_sim ** 2).mean()
    return loss