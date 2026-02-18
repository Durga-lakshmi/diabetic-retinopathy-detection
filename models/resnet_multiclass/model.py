import torch.nn as nn
from torchvision import models

class ResNet18Multiclass(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=False):
        super().__init__()

        # Load pretrained backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Optionally freeze backbone (feature extraction)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace FC head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 5)

    def forward(self, x):
        return self.model(x)

# class ResNet18Multiclass(nn.Module):
#     def __init__(self, num_classes=5, pretrained=True, freeze_early_layers=False):
#         super().__init__()

#         # Load pretrained ResNet-18 backbone
#         weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#         self.model = models.resnet18(weights=weights)

#         # Freeze early layers (conv1 + layer1)
#         if freeze_early_layers:
#             for param in self.model.conv1.parameters():
#                 param.requires_grad = False
#             for param in self.model.layer1.parameters():
#                 param.requires_grad = False

#         # Replace the FC head with the desired number of classes
#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)