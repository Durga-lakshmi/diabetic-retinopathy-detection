import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

# ------------------ Dense121 ------------------
class Dense121(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ------------------ load cfg ------------------
        self.num_classes = cfg.num_classes
        self.growth_rate = cfg.growth_rate
        self.block_config = cfg.block_config
        self.num_init_features = cfg.num_init_features
        self.bn_size = cfg.bn_size
        self.drop_rate = cfg.drop_rate

        # initial conv
        self.conv1 = nn.Conv2d(3, self.num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks
        num_channels = self.num_init_features
        self.features = nn.Sequential()
        for i, num_layers in enumerate(self.block_config):
            block = DenseBlock(num_layers, num_channels, self.growth_rate, self.bn_size, self.drop_rate)
            self.features.add_module(f"denseblock{i+1}", block)
            num_channels += num_layers * self.growth_rate

            if i != len(self.block_config) - 1:
                out_channels = num_channels // 2
                self.features.add_module(f"transition{i+1}", Transition(num_channels, out_channels))
                num_channels = out_channels

        # final batch norm
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.classifier = nn.Linear(num_channels, self.num_classes)

        # ------------------ weight init ------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.features(x)
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits  # logits ，BCEWithLogitsLoss 



# ------------------ Dense Layer ------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = growth_rate * bn_size
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = torch.cat([x, out], dim=1)
        return out

# ------------------ Dense Block ------------------
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ------------------ Transition ------------------
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


def get_densenet121(num_classes):
    model = densenet121(weights="IMAGENET1K_V1")   # load ImageNet weights
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

