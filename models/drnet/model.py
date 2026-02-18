# import torch
# import torch.nn as nn
# # from torchsummary import summary

# class DRNet(nn.Module):
#     def __init__(self, num_classes=1):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),      # 128x128

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),      # 64x64

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),      # 32x32

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4))  # 256×4×4
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes),
#             #nn.Sigmoid()  # add for binary classification
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x



import torch
import torch.nn as nn

class DRNet(nn.Module):
    """Small CNN for data-starved training (lightweight, less overfit)."""
    def __init__(self, num_classes=1):
        super().__init__()
        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.enc = nn.Sequential(
            C(3, 32),
            C(32, 32),
            nn.MaxPool2d(2),        # 128

            C(32, 64),
            C(64, 64),
            nn.MaxPool2d(2),        # 64

            C(64, 128),
            C(128, 128),
            nn.MaxPool2d(2),        # 32

            C(128, 256),
            nn.AdaptiveAvgPool2d(1) # 1x1
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)   # logits -> BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.fc(x)
        return x

# # Training
# epochs: 80
# lr: 0.0001
# # Logging
# log_interval: 5
# eval_interval: 1

# #save_path: ./checkpoints/dense/best_model_aug_bs64_11_21.pth
# save_path: artifacts/best_model.pth

# # Early Stopping
# patience: 8

# #weight decay
# weight_decay: 1e-3
import torch
import torch.nn as nn
# from torchsummary import summary

class DRNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 128x128

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64x64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32x32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 256×4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # add for binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Print summary
# print(summary(model, (3, 256, 256)))