import torch
import torch.nn as nn

# Global Perceptron Module
class GlobalPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branch 1: split & BN + FC + ReLU + FC
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        # Branch 2: avgpool → BN + FC + ReLU + FC
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.fc3 = nn.Linear(in_channels, out_channels)
        self.fc4 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        # Branch 1: flatten spatially
        x1 = x.permute(0,2,3,1).reshape(-1, c)
        x1 = self.fc2(torch.relu(self.fc1(self.bn1(x).mean(dim=[2,3]))))
        # Branch 2: avgpool + BN + FC + FC
        x2 = self.pool(x).view(b, c)
        x2 = self.fc4(torch.relu(self.fc3(self.bn2(x2))))
        # broadcast and combine branches
        x1 = x1.view(b, -1)
        out = x1 + x2
        return out

# Partition Perceptron Module
class PartitionPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x

# Local Perceptron Module (dilated convolutions)
class LocalPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d),
                nn.BatchNorm2d(out_channels)
            ) for d in [1,2,5]
        ])

    def forward(self, x):
        out = sum([conv(x) for conv in self.dilated_convs])
        return out

# DMLP Block (three modules + fusion)
class DMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_perc = GlobalPerceptron(in_channels, out_channels)
        self.partition_perc = PartitionPerceptron(in_channels, out_channels)
        self.local_perc = LocalPerceptron(in_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        g = self.global_perc(x)
        p = self.partition_perc(x)
        l = self.local_perc(x)
        fusion = p + l
        fusion = self.bn(fusion)
        # Residual addition
        fusion = fusion + self.shortcut(x)
        fusion = fusion + g.unsqueeze(-1).unsqueeze(-1)
        fusion = torch.relu(fusion)
        return fusion

# Main DMLPFFN network
class DMLPFFN(nn.Module):
    def __init__(self, in_channels=128, num_classes=4):
        super().__init__()
        # Low-level branch
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DMLPBlock(16, 16),
            nn.Conv2d(16, 64, 1)
        )
        # Middle-level branch
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DMLPBlock(32, 32),
            nn.Conv2d(32, 64, 1)
        )
        # High-level branch
        self.high_branch = nn.Sequential(
            DMLPBlock(in_channels, 64),
            nn.Conv2d(64, 64, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, channels/bands after PCA, height, width)
        low = self.low_branch(x)
        mid = self.mid_branch(x)
        high = self.high_branch(x)
        fused = low + mid + high
        pooled = self.gap(fused).flatten(1)
        out = self.classifier(pooled)
        return out
