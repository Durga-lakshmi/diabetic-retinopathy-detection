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




##

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvNeXt(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ------------------ load cfg ------------------
        self.num_classes = cfg.num_classes          # number of output classes
        self.variant = cfg.model_size               # tiny/small/base/large
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

        # ------------------ change classifier ------------------
        in_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_dim, self.num_classes)

        # ====== 拆出主干结构，后面要从中拿 feature map ======
        self.features = self.backbone.features      # (B, C, Hf, Wf)
        self.avgpool = self.backbone.avgpool        # AdaptiveAvgPool2d
        self.classifier = self.backbone.classifier  # LayerNorm + Linear

        # ====== heatmap 头：从 C 通道 → 1 通道 ======
        self.heatmap_head = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, x, return_heatmap: bool = False):
        """
        默认行为和原来一致：return_heatmap=False 时只返回 logits
        如果 return_heatmap=True，则返回 (logits, heatmap)
        """
        # backbone feature map
        feats = self.features(x)            # (B, C, Hf, Wf)

        # 分类分支（保持和 torchvision 一致）
        pooled = self.avgpool(feats)        # (B, C, 1, 1)
        #pooled = pooled.flatten(1)          # (B, C)
        logits = self.classifier(pooled)    # (B, num_classes)

        if not return_heatmap:
            return logits

        # heatmap 分支：C → 1
        heatmap = self.heatmap_head(feats)  # (B, 1, Hf, Wf)

        # 如需和输入同大小再做可视化，可以在外面再插值：
        # heatmap_up = F.interpolate(heatmap, size=x.shape[2:], mode="bilinear", align_corners=False)
        # return logits, heatmap_up

        return logits, heatmap


