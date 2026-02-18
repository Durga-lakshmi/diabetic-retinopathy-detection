import torch
import torch.nn as nn
from torchvision import models


class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        if pretrained:
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
        else:
            self.model = models.resnet18(weights=None)

        # Replace final FC for binary classification (single logit)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)

# class ResNet18Binary(nn.Module):
#     def __init__(self, pretrained=True, freeze_backbone=True):
#         super().__init__()
#         # Load ResNet18
#         self.model = models.resnet18(
#             weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#         )

#         # Replace final FC for binary classification (1 logit)
#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(in_features, 1)

#         # Freeze backbone if requested
#         if freeze_backbone:
#             # Freeze all layers except the last block (layer4) and fc
#             for name, param in self.model.named_parameters():
#                 if "layer4" not in name and "fc" not in name:
#                     param.requires_grad = False

#     def forward(self, x):
#         return self.model(x)


import torch
import torch.nn as nn
from torchvision import models


class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        if pretrained:
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
        else:
            self.model = models.resnet18(weights=None)

        # Replace final FC for binary classification (single logit)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)

# class ResNet18Binary(nn.Module):
#     def __init__(self, pretrained=True, freeze_backbone=True):
#         super().__init__()
#         # Load ResNet18
#         self.model = models.resnet18(
#             weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#         )

#         # Replace final FC for binary classification (1 logit)
#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(in_features, 1)

#         # Freeze backbone if requested
#         if freeze_backbone:
#             # Freeze all layers except the last block (layer4) and fc
#             for name, param in self.model.named_parameters():
#                 if "layer4" not in name and "fc" not in name:
#                     param.requires_grad = False

#     def forward(self, x):
#         return self.model(x)