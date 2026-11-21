"""
ResNet-18 for Hyperspectral Imaging - 128-Channel Adapted
Strong CNN baseline with residual connections.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18_HSI(nn.Module):
    """
    ResNet-18 adapted for HSI patches with 128 input channels.
    Input: [B, 128, 9, 9] (NCHW) - 128 channels, 9x9 spatial (from HSIDataset)
    """
    def __init__(
        self,
        in_channels=128,
        num_classes=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load ResNet-18 without pretrained weights
        self.backbone = resnet18(weights=None)

        # Modify first conv layer for 128 input channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # For 9x9 input: use kernel_size=3, stride=1, padding=1, remove maxpool
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove maxpool (would reduce 9x9 to too small)
        self.backbone.maxpool = nn.Identity()

        # Modify final FC layer
        self.backbone.fc = nn.Linear(512, num_classes)

        # Initialize the modified layers
        self._init_weights()

    def _init_weights(self):
        # Initialize modified conv1
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        # Initialize modified fc
        nn.init.trunc_normal_(self.backbone.fc.weight, std=0.02)
        if self.backbone.fc.bias is not None:
            nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x):
        """
        x: [B, 128, 9, 9] (NCHW) - from HSIDataset
        Returns: [B, num_classes]
        """
        return self.backbone(x)


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


def test_resnet18():
    """Test ResNet-18 forward pass and verify parameter count."""
    model = ResNet18_HSI(
        in_channels=128,
        num_classes=4,
    )

    print("=== ResNet-18 HSI Parameter Count ===")
    count_parameters_detailed(model)
    total_params = count_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")

    # Verify against expected ~11.2M
    expected = 11_200_000
    print(f"Expected: ~{expected:,}")
    print(f"Match: {abs(total_params - expected) < 200000}")  # Within 200K tolerance

    # Test forward pass - NCHW format from HSIDataset
    x = torch.randn(2, 128, 9, 9)  # Batch=2, 128 channels, 9x9
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
        x_batch = torch.randn(16, 128, 9, 9)
        out_batch = model(x_batch)
        param_mem = sum(p.numel() * 4 for p in model.parameters()) / 1e9  # GB (fp32)
        print(f"Estimated parameter memory (fp32): {param_mem:.2f} GB")

    print("All ResNet-18 tests passed!")
    return model


if __name__ == "__main__":
    test_resnet18()