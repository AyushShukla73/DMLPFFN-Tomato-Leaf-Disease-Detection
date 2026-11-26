"""Parameter counting script for DMLPFFN implementations - avoiding quote issues."""
import torch
import torch.nn as nn

# =============================================================================
# Implementation 1: dmlpffn_disease.py (Original / Current)
# =============================================================================

class GlobalPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.fc3 = nn.Linear(in_channels, out_channels)
        self.fc4 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = x.permute(0,2,3,1).reshape(-1, c)
        x1 = self.fc2(torch.relu(self.fc1(self.bn1(x).mean(dim=[2,3]))))
        x2 = self.pool(x).view(b, c)
        x2 = self.fc4(torch.relu(self.fc3(self.bn2(x2))))
        x1 = x1.view(b, -1)
        out = x1 + x2
        return out


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


class PartitionPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


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
        fusion = fusion + self.shortcut(x)
        fusion = fusion + g.unsqueeze(-1).unsqueeze(-1)
        fusion = torch.relu(fusion)
        return fusion


class DMLPFFN(nn.Module):
    def __init__(self, in_channels=128, num_classes=4, dropout=0.7):
        super().__init__()
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DMLPBlock(16, 16),
            nn.Conv2d(16, 64, 1)
        )
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DMLPBlock(32, 32),
            nn.Conv2d(32, 64, 1)
        )
        self.high_branch = nn.Sequential(
            DMLPBlock(in_channels, 64),
            nn.Conv2d(64, 64, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        low = self.low_branch(x)
        mid = self.mid_branch(x)
        high = self.high_branch(x)
        fused = low + mid + high
        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out


# =============================================================================
# Implementation 2: Legacy baseline_models.py (Proper concat)
# =============================================================================

class LocalPerceptronAblation(nn.Module):
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


class DMLPBlockAblation(nn.Module):
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

        if fusion_type == 'concat':
            self.fused_channels = sum([
                out_channels if use_global else 0,
                out_channels if use_partition else 0,
                out_channels if use_local else 0
            ])
        else:
            self.fused_channels = out_channels

        self.shortcut = nn.Conv2d(in_channels, self.fused_channels, 1)
        self.bn = nn.BatchNorm2d(self.fused_channels)

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

        b, c, h, w = x.shape

        processed_outputs = []
        for name, out in outputs:
            if out.dim() == 2:
                out = out.unsqueeze(-1).unsqueeze(-1).expand(b, -1, h, w).contiguous()
            processed_outputs.append(out)

        if self.fusion_type == 'concat':
            fusion = torch.cat(processed_outputs, dim=1)
        else:
            fusion = sum(processed_outputs)

        fusion = self.bn(fusion)
        fusion = torch.relu(fusion)

        shortcut_out = self.shortcut(x)
        fusion = fusion + shortcut_out
        fusion = torch.relu(fusion)
        return fusion


class DMLPFFNAblation(nn.Module):
    def __init__(self, in_channels=128, num_classes=4, dropout=0.7,
                 use_global=True, use_partition=True, use_local=True,
                 fusion_type='add', dilation_rates=[1,2,5]):
        super().__init__()
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_type = fusion_type
        self.dilation_rates = dilation_rates

        low_out = 16 if fusion_type != 'concat' else (sum([16 if use_global else 0, 16 if use_partition else 0, 16 if use_local else 0]))
        mid_out = 32 if fusion_type != 'concat' else (sum([32 if use_global else 0, 32 if use_partition else 0, 32 if use_local else 0]))
        high_out = 64 if fusion_type != 'concat' else (sum([64 if use_global else 0, 64 if use_partition else 0, 64 if use_local else 0]))

        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(16, 16, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(low_out, 64, 1)
        )
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(32, 32, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(mid_out, 64, 1)
        )
        self.high_branch = nn.Sequential(
            DMLPBlockAblation(in_channels, 64, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(high_out, 64, 1)
        )

        if fusion_type == 'concat':
            fused_channels = 192
        else:
            fused_channels = 64

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_channels, num_classes)

    def forward(self, x):
        low = self.low_branch(x)
        mid = self.mid_branch(x)
        high = self.high_branch(x)

        if self.fusion_type == 'concat':
            fused = torch.cat([low, mid, high], dim=1)
        else:
            fused = low + mid + high

        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out


# =============================================================================
# Implementation 3: Historical buggy baseline_models.py (concat bug)
# =============================================================================

class DMLPFFNAblationBuggy(nn.Module):
    """Historical buggy version where concat is overwritten."""
    def __init__(self, in_channels=128, num_classes=4, dropout=0.7,
                 use_global=True, use_partition=True, use_local=True,
                 fusion_type='add', dilation_rates=[1,2,5]):
        super().__init__()
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_type = fusion_type
        self.dilation_rates = dilation_rates

        low_out = 16 if fusion_type != 'concat' else (sum([16 if use_global else 0, 16 if use_partition else 0, 16 if use_local else 0]))
        mid_out = 32 if fusion_type != 'concat' else (sum([32 if use_global else 0, 32 if use_partition else 0, 32 if use_local else 0]))
        high_out = 64 if fusion_type != 'concat' else (sum([64 if use_global else 0, 64 if use_partition else 0, 64 if use_local else 0]))

        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(16, 16, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(low_out, 64, 1)
        )
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(32, 32, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(mid_out, 64, 1)
        )
        self.high_branch = nn.Sequential(
            DMLPBlockAblation(in_channels, 64, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(high_out, 64, 1)
        )

        # BUG: Always uses 64 channels regardless of fusion_type
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
            # BUG: concat result immediately overwritten
            fused = torch.cat([low, mid, high], dim=1)
            fused = low + mid + high  # Bug line
        else:
            fused = low + mid + high

        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("=" * 80)
    print("PARAMETER COUNTS FOR DMLPFFN IMPLEMENTATIONS")
    print("=" * 80)

    # Test config
    configs = {
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

    x = torch.randn(2, 128, 9, 9)

    print("\n1. dmlpffn_disease.py (Original / Current)")
    print("-" * 40)
    model = DMLPFFN(in_channels=128, num_classes=4, dropout=0.7)
    total, _ = count_parameters(model)
    print(f"  full (additive only): {total:>10,} params")
    with torch.no_grad():
        out = model(x)
    print(f"  Output shape: {tuple(out.shape)}")

    print("\n2. Legacy baseline_models.py (Proper concat)")
    print("-" * 40)
    for name, config in configs.items():
        model = DMLPFFNAblation(
            in_channels=128, num_classes=4, dropout=0.7,
            use_global=config['use_global'], use_partition=config['use_partition'],
            use_local=config['use_local'], fusion_type=config['fusion_type'],
            dilation_rates=config['dilation_rates']
        )
        total, _ = count_parameters(model)
        with torch.no_grad():
            out = model(x)
        if name == 'fusion_concat':
            print(f"  {name:20s}: {total:>10,} params (PROPER concat)")
        else:
            print(f"  {name:20s}: {total:>10,} params")

    print("\n3. Historical buggy baseline_models.py (concat bug)")
    print("-" * 40)
    for name, config in configs.items():
        model = DMLPFFNAblationBuggy(
            in_channels=128, num_classes=4, dropout=0.7,
            use_global=config['use_global'], use_partition=config['use_partition'],
            use_local=config['use_local'], fusion_type=config['fusion_type'],
            dilation_rates=config['dilation_rates']
        )
        total, _ = count_parameters(model)
        with torch.no_grad():
            out = model(x)
        if name == 'fusion_concat':
            print(f"  {name:20s}: {total:>10,} params (BUG - concat overwritten)")
        else:
            print(f"  {name:20s}: {total:>10,} params")

    print("\n" + "=" * 80)

    # Summary table
    print("\nSUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Config':<20} {'Legacy (proper)':>18} {'Historical (buggy)':>18}")
    print("-" * 80)

    for name, config in configs.items():
        model1 = DMLPFFNAblation(
            in_channels=128, num_classes=4, dropout=0.7,
            use_global=config['use_global'], use_partition=config['use_partition'],
            use_local=config['use_local'], fusion_type=config['fusion_type'],
            dilation_rates=config['dilation_rates']
        )
        model2 = DMLPFFNAblationBuggy(
            in_channels=128, num_classes=4, dropout=0.7,
            use_global=config['use_global'], use_partition=config['use_partition'],
            use_local=config['use_local'], fusion_type=config['fusion_type'],
            dilation_rates=config['dilation_rates']
        )
        p1, _ = count_parameters(model1)
        p2, _ = count_parameters(model2)
        print(f"{name:<20} {p1:>18,} {p2:>18,}")
    print("-" * 80)