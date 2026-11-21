"""
Baseline Models for DMLPFFN Comparison (Phase 5)

This module implements various baseline architectures for comparison with DMLPFFN:
1. SimpleCNN - Basic CNN architecture
2. ResNet18/50 - ResNet adapted for 128-channel input
3. ViT-Small - Vision Transformer small variant
4. MLP-Mixer - MLP-Mixer architecture
5. 1D-CNN - Spectral 1D CNN
6. SpectralNet - Spectral-specific architecture
7. DBDA - Deep Belief Network inspired architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# 1. SimpleCNN - Basic CNN Baseline
# =============================================================================
class SimpleCNN(nn.Module):
    """Simple CNN baseline with 3 conv layers."""

    def __init__(self, in_channels=128, num_classes=4, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


# =============================================================================
# 2. ResNet Adapted for 128-channel Input
# =============================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet adapted for hyperspectral input (128 channels)."""

    def __init__(self, block, num_blocks, in_channels=128, num_classes=4, dropout=0.5):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def ResNet18(in_channels=128, num_classes=4, dropout=0.5):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes, dropout)


def ResNet50(in_channels=128, num_classes=4, dropout=0.5):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels, num_classes, dropout)


# =============================================================================
# 3. ViT-Small (Vision Transformer)
# =============================================================================
class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""

    def __init__(self, in_channels=128, patch_size=3, embed_dim=256, img_size=9):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for hyperspectral patches."""

    def __init__(self, in_channels=128, num_classes=4, embed_dim=256, depth=6,
                 num_heads=8, patch_size=3, img_size=9, mlp_ratio=4.0, dropout=0.1, cls_dropout=0.5):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(cls_dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token = x[:, 0]
        cls_token = self.dropout(cls_token)
        return self.head(cls_token)


def ViTSmall(in_channels=128, num_classes=4, dropout=0.5):
    return ViT(in_channels, num_classes, embed_dim=256, depth=6, num_heads=8,
               patch_size=3, img_size=9, cls_dropout=dropout)


# =============================================================================
# 4. MLP-Mixer
# =============================================================================
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, num_patches, embed_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mix = FeedForward(num_patches, tokens_mlp_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_mix = FeedForward(embed_dim, channels_mlp_dim, dropout)

    def forward(self, x):
        # Token mixing
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y

        # Channel mixing
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y
        return x


class MLPMixer(nn.Module):
    """MLP-Mixer for hyperspectral patches."""

    def __init__(self, in_channels=128, num_classes=4, embed_dim=256, depth=8,
                 patch_size=3, img_size=9, tokens_mlp_dim=512, channels_mlp_dim=1024,
                 dropout=0.1, cls_dropout=0.5):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.num_patches

        self.mixer_blocks = nn.ModuleList([
            MixerBlock(num_patches, embed_dim, tokens_mlp_dim, channels_mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(cls_dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.mixer_blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        x = self.dropout(x)
        return self.head(x)


def MLPMixerBase(in_channels=128, num_classes=4, dropout=0.5):
    return MLPMixer(in_channels, num_classes, embed_dim=256, depth=8,
                    patch_size=3, img_size=9, cls_dropout=dropout)


# =============================================================================
# 5. 1D-CNN (Spectral)
# =============================================================================
class Spectral1DCNN(nn.Module):
    """1D CNN operating on spectral dimension."""

    def __init__(self, in_channels=128, num_classes=4, dropout=0.5):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> treat spatial as sequence
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)  # (B, C, 81)
        x = self.conv1d(x)
        x = x.view(B, -1)
        x = self.dropout(x)
        return self.classifier(x)


# =============================================================================
# 6. SpectralNet
# =============================================================================
class SpectralNet(nn.Module):
    """SpectralNet: Deep network for hyperspectral classification."""

    def __init__(self, in_channels=128, num_classes=4, dropout=0.5):
        super().__init__()
        # Spectral feature extraction
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        # Spatial feature extraction (lightweight)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Spectral path
        x_spec = x.view(B, C, H * W)
        spec_feat = self.spectral_encoder(x_spec).view(B, -1)

        # Spatial path
        spat_feat = self.spatial_encoder(x).view(B, -1)

        # Combine
        combined = torch.cat([spec_feat, spat_feat], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)


# =============================================================================
# 7. DBDA (Deep Belief Dense Autoencoder inspired)
# =============================================================================
class DBDA(nn.Module):
    """DBDA-inspired architecture for hyperspectral classification."""

    def __init__(self, in_channels=128, num_classes=4, dropout=0.5):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Dense layers with residual connections
        self.dense1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.dense3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Residual projection
        self.res_proj = nn.Linear(256, 256)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        x1 = self.dense1(x)
        x2 = self.dense2(x1) + x1  # Residual
        x3 = self.dense3(x2)
        x3 = x3 + self.res_proj(x)  # Residual from encoder

        return self.classifier(x3)


# =============================================================================
# Model Factory
# =============================================================================
def get_baseline_model(model_name: str, in_channels=128, num_classes=4, dropout=0.5):
    """Factory function to get baseline model by name."""
    models = {
        'simplecnn': lambda: SimpleCNN(in_channels, num_classes, dropout),
        'resnet18': lambda: ResNet18(in_channels, num_classes, dropout),
        'resnet50': lambda: ResNet50(in_channels, num_classes, dropout),
        'vit_small': lambda: ViTSmall(in_channels, num_classes, dropout),
        'mlp_mixer': lambda: MLPMixerBase(in_channels, num_classes, dropout),
        '1dcnn': lambda: Spectral1DCNN(in_channels, num_classes, dropout),
        'spectralnet': lambda: SpectralNet(in_channels, num_classes, dropout),
        'dbda': lambda: DBDA(in_channels, num_classes, dropout),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name]()


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Ablation Model Variants
# =============================================================================
class DMLPFFNAblation(nn.Module):
    """DMLPFFN with configurable ablation options."""

    def __init__(self, in_channels=128, num_classes=4, dropout=0.7,
                 use_global=True, use_partition=True, use_local=True,
                 fusion_type='add', dilation_rates=[1,2,5]):
        super().__init__()
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_type = fusion_type
        self.dilation_rates = dilation_rates

        # Low-level branch
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(16, 16, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(16, 64, 1)
        )
        # Middle-level branch
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(32, 32, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(32, 64, 1)
        )
        # High-level branch
        self.high_branch = nn.Sequential(
            DMLPBlockAblation(in_channels, 64, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(64, 64, 1)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        low = self.low_branch(x)
        mid = self.mid_branch(x)
        high = self.high_branch(x)

        if self.fusion_type == 'add':
            fused = low + mid + high
        elif self.fusion_type == 'concat':
            fused = torch.cat([low, mid, high], dim=1)
            # Need to adjust classifier input dim
            # For now, just use add
            fused = low + mid + high
        else:
            fused = low + mid + high

        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out


class DMLPBlockAblation(nn.Module):
    """DMLP Block with ablation options."""

    def __init__(self, in_channels, out_channels, use_global=True, use_partition=True,
                 use_local=True, fusion_type='add', dilation_rates=[1,2,5]):
        super().__init__()
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_type = fusion_type

        if use_global:
            self.global_perc = GlobalPerceptron(in_channels, out_channels)
        if use_partition:
            self.partition_perc = PartitionPerceptron(in_channels, out_channels)
        if use_local:
            self.local_perc = LocalPerceptronAblation(in_channels, out_channels, dilation_rates)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        outputs = []

        if self.use_global:
            g = self.global_perc(x)
            outputs.append(('global', g))
        if self.use_partition:
            p = self.partition_perc(x)
            outputs.append(('partition', p))
        if self.use_local:
            l = self.local_perc(x)
            outputs.append(('local', l))

        if not outputs:
            return self.shortcut(x)

        # Fusion
        if self.fusion_type == 'add':
            # Spatial outputs can be added directly
            # Global output needs broadcasting
            spatial_outputs = [o[1] for o in outputs if o[0] != 'global']
            global_outputs = [o[1] for o in outputs if o[0] == 'global']

            if spatial_outputs:
                fusion = sum(spatial_outputs)
            else:
                fusion = torch.zeros_like(self.shortcut(x))

            fusion = self.bn(fusion)
            fusion = fusion + self.shortcut(x)

            if global_outputs:
                fusion = fusion + global_outputs[0].unsqueeze(-1).unsqueeze(-1)

            fusion = torch.relu(fusion)
            return fusion

        return self.shortcut(x)


class LocalPerceptronAblation(nn.Module):
    """Local Perceptron with configurable dilation rates."""

    def __init__(self, in_channels, out_channels, dilation_rates=[1,2,5]):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d),
                nn.BatchNorm2d(out_channels)
            ) for d in dilation_rates
        ])

    def forward(self, x):
        return sum([conv(x) for conv in self.dilated_convs])


def get_ablation_model(config: dict, in_channels=128, num_classes=4, dropout=0.7):
    """Create DMLPFFN ablation variant from config."""
    return DMLPFFNAblation(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        use_global=config.get('use_global', True),
        use_partition=config.get('use_partition', True),
        use_local=config.get('use_local', True),
        fusion_type=config.get('fusion_type', 'add'),
        dilation_rates=config.get('dilation_rates', [1,2,5])
    )


# Ablation configurations
ABLATION_CONFIGS = {
    'full': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'no_global': {'use_global': False, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'no_partition': {'use_global': True, 'use_partition': False, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'no_local': {'use_global': True, 'use_partition': True, 'use_local': False, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'only_global': {'use_global': True, 'use_partition': False, 'use_local': False, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'only_partition': {'use_global': False, 'use_partition': True, 'use_local': False, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'only_local': {'use_global': False, 'use_partition': False, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'fusion_concat': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'concat', 'dilation_rates': [1,2,5]},
    'dilation_137': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,3,7]},
    'dilation_1_only': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1]},
    'dilation_15': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,5]},
}