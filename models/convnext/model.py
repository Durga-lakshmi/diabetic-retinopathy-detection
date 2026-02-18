
import torch
import torch.nn as nn
import torchvision.models as models

class ConvNeXt(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ------------------ load cfg ------------------
        self.num_classes = cfg.num_classes          # number of output classes
        self.variant = cfg.model_size                  # tiny/small/base/large
        self.pretrained = cfg.pretrained            # True / False

        # ------------------ choice size of model ------------------
        variant_map = {
            "tiny": models.convnext_tiny,
            "small": models.convnext_small,
            "base": models.convnext_base,
            "large": models.convnext_large
        }

        if self.variant not in variant_map:
            raise ValueError(f"Unknown ConvNeXt variant {self.variant}")

        # ------------------ load backbone ------------------
        if self.pretrained:
            self.backbone = variant_map[self.variant](weights="DEFAULT")
        else:
            self.backbone = variant_map[self.variant](weights=None)

        # ------------------ change classifer ------------------
        in_dim = self.backbone.classifier[-1].in_features  #classifier[2]
        self.backbone.classifier[-1] = nn.Linear(in_dim, self.num_classes) #




    def forward(self, x):
        return self.backbone(x)
