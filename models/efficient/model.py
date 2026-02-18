import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.variant = cfg.model_size   # b0/b1/b2...
        self.pretrained = cfg.pretrained

        variant_map = {
            "b0": models.efficientnet_b0,
            "b1": models.efficientnet_b1,
            "b2": models.efficientnet_b2,
            "b3": models.efficientnet_b3,
            "b4": models.efficientnet_b4,
            "b5": models.efficientnet_b5,
            "b6": models.efficientnet_b6,
            "b7": models.efficientnet_b7,
        }

        if self.variant not in variant_map:
            raise ValueError(f"Unknown EfficientNet variant {self.variant}")

        # load backbone
        if self.pretrained:
            self.backbone = variant_map[self.variant](weights="DEFAULT")
        else:
            self.backbone = variant_map[self.variant](weights=None)

        # change classifier
        in_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_dim, self.num_classes)

    def forward(self, x):
        return self.backbone(x)
