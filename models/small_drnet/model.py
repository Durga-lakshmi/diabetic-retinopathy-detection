import torch
import torch.nn as nn

class SmallDRNet(nn.Module):
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
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)   # logits -> BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.fc(x)
        return x


# defaults:
#   - dataset: idrid  # Options: idrid, eyepacs
#   - model: small_drnet #drnet_improved #basic_cnn #dense121
#   - _self_
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

# mean: [-0.2259, -1.0937, -1.4864]
# std: [1.2921, 0.6861, 0.2452]

# dropout_rate: 0.4
