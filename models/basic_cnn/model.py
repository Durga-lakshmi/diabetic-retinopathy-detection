import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.depth

        conv_layers = [nn.Conv2d(3, 4, kernel_size=5, padding='same')]
        for i in range(self.depth):
            conv_layers.append(ConvBlock(4 * (2 ** i), cfg.kernel_size))

        self.conv_layers = nn.ModuleList(conv_layers)

        self.lin_1 = nn.Linear(4 * 2 ** self.depth, 32)
        self.lin_2 = nn.Linear(32, 8)
        self.lin_3 = nn.Linear(8, 1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        # Global Average Pooling
        x = x.mean(dim=(-1, -2))

        x = self.lin_1(x)
        x = F.relu(x)

        x = self.lin_2(x)
        x = F.relu(x)

        x = self.lin_3(x)
        x = F.sigmoid(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, c_in, kernel_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_in, kernel_size, padding='same')
        self.conv_2 = nn.Conv2d(c_in,
                                2 * c_in,
                                kernel_size,
                                stride=2,
                                padding=kernel_size // 2)

        self.norm = nn.BatchNorm2d(c_in)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)

        x = self.norm(x)

        x = self.conv_2(x)
        x = F.relu(x)

        return x
