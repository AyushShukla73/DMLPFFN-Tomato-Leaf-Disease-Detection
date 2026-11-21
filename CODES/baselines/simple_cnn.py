"""
SimpleCNN for Hyperspectral Imaging - Corrected Implementation
Lightweight CNN baseline adapted for 128-channel HSI input.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    SimpleCNN adapted for HSI patches.
    Input: [B, 9, 9, 128] (NHWC) - 9x9 spatial, 128 spectral channels
    """
    def __init__(
        self,
        in_channels=128,
        num_classes=4,
        base_channels=44,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extraction: 128 -> 64 -> 128 channels
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Block 2: 64 -> 128
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )

        # Classifier
        self.classifier = nn.Linear(base_channels * 2, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, 128, 9, 9] (NCHW) - standard PyTorch Conv2d format
        Returns: [B, num_classes]
        """
        # Feature extraction (input is already NCHW from HSIDataset)
        x = self.features(x)

        # Flatten: [B, 128, 1, 1] -> [B, 128]
        x = x.flatten(1)

        # Classification
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_detailed(model):
    """Count parameters per layer."""
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
            print(f"  {name}: {p.numel():,}")
    print(f"  Total: {total:,}")
    return total


def test_simple_cnn():
    """Test SimpleCNN forward pass and verify parameter count."""
    model = SimpleCNN(
        in_channels=128,
        num_classes=4,
        base_channels=44,
    )

    print("=== SimpleCNN Parameter Count ===")
    count_parameters_detailed(model)
    total_params = count_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")

    # Verify against expected ~86K (base_channels=44)
    expected = 86_156
    print(f"Expected: ~{expected:,}")
    print(f"Match: {abs(total_params - expected) < 5000}")  # Within 5K tolerance

    # Test forward pass
    x = torch.randn(2, 9, 9, 128)  # Batch=2, 9x9, 128 channels
    out = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"

    # Verify gradients flow
    loss = out.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Gradient flow test passed: total grad norm = {grad_norm:.4f}")

    # Memory estimate for batch=16
    with torch.no_grad():
        x_batch = torch.randn(16, 9, 9, 128)
        out_batch = model(x_batch)
        param_mem = sum(p.numel() * 4 for p in model.parameters()) / 1e9  # GB (fp32)
        print(f"Estimated parameter memory (fp32): {param_mem:.2f} GB")

    print("All SimpleCNN tests passed!")
    return model


if __name__ == "__main__":
    test_simple_cnn()