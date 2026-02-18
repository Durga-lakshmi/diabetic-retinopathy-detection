import torch
import torch.nn as nn
import torch.nn.functional as F



#-----------------------------------------------------------
#!!! ensemble learning - feature level


class FeatureExtractor(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        self.model = model
        self.model_type = model_type.lower()

        # -------------------------
        # Feature dimension
        # -------------------------

        if self.model_type == "convnext":
            # classifier = [LayerNorm2d, Flatten, Linear]
            self.feature_dim = model.backbone.classifier[-1].in_features

            # classifier -> Identity
            model.backbone.classifier[-1] = nn.Identity()

        elif self.model_type == "dense121":
            self.feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()

        elif self.model_type == "efficientnet":
            # efficientnet.classifier = [Dropout, Linear]
            self.feature_dim = model.backbone.classifier[1].in_features
            model.backbone.classifier[1] = nn.Identity()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    # ----------------------------------------------------
    # Forward path — over forward_features
    # ----------------------------------------------------
    def forward(self, x):

        # =========================
        # 1) ConvNeXt (torchvision)
        # =========================
        if self.model_type == "convnext":
            # 1) backbone.features
            feat = self.model.backbone.features(x)
            # 2) backbone.avgpool
            feat = self.model.backbone.avgpool(feat)
            # 3) flatten
            feat = feat.flatten(1)
            return feat

        # =========================
        # 2) DenseNet
        # =========================
        if self.model_type == "dense121":
            feat = self.model.features(x)
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return feat

        # =========================
        # 3) EfficientNet
        # =========================
        if self.model_type == "efficientnet":
            feat = self.model.backbone.features(x)
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return feat

        raise ValueError("Unknown model type")




# -----------------------------
# Fusion Head
# -----------------------------
class FusionHead(nn.Module):
    def __init__(self, total_dim, num_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# -----------------------------
# Fusion Model
# -----------------------------
class FusionModel(nn.Module):
    def __init__(self, extractors, num_classes=5):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)
        total_dim = sum([e.feature_dim for e in extractors])
        self.head = FusionHead(total_dim, num_classes)

    def forward(self, x):
        feats = [e(x) for e in self.extractors]
        f = torch.cat(feats, dim=1)
        return self.head(f)


class AttentionFusionHead(nn.Module):
    """
    Attention-based feature fusion for 3 backbones (with projection).
    """
    def __init__(self, dims, fusion_dim=512, num_classes=5):
        super().__init__()

        self.dims = dims
        self.fusion_dim = fusion_dim

        # 1. projection: unify feature dimensions
        self.proj1 = nn.Linear(dims[0], fusion_dim)
        self.proj2 = nn.Linear(dims[1], fusion_dim)
        self.proj3 = nn.Linear(dims[2], fusion_dim)

        # 2. attention network
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3),          # 3 backbones
            nn.Softmax(dim=1)
        )

        # 3. classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, f1, f2, f3,return_feats=False):
        # ---- projection ----
        f1 = self.proj1(f1)    # [B, fusion_dim]
        f2 = self.proj2(f2)
        f3 = self.proj3(f3)

        # ---- attention ----
        concat = torch.cat([f1, f2, f3], dim=1)  # [B, fusion_dim*3]
        attn = self.attention(concat)             # [B, 3]

        a1 = attn[:, 0].unsqueeze(1)
        a2 = attn[:, 1].unsqueeze(1)
        a3 = attn[:, 2].unsqueeze(1)

        # ---- weighted fusion ----
        fused = a1 * f1 + a2 * f2 + a3 * f3       # [B, fusion_dim]

        logits = self.classifier(fused)

        if return_feats:
            # 返回 logits 以及 projection 后的特征列表
            return logits, [f1, f2, f3]

        return logits


class AttentionFusionModel(nn.Module):
    def __init__(self, extractors, num_classes=5):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)

        dims = [e.feature_dim for e in extractors]
        self.head = AttentionFusionHead(dims, num_classes=num_classes)

    def forward(self, x, return_feats: bool = False):
        f1 = self.extractors[0](x)
        f2 = self.extractors[1](x)
        f3 = self.extractors[2](x)
        return self.head(f1, f2, f3, return_feats=return_feats)

