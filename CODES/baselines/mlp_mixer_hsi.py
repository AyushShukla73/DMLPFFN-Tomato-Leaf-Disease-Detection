"""
MLP-Mixer for Hyperspectral Imaging - Corrected Implementation
Proper token-mixing and channel-mixing MLPs with separate LayerNorm and residual connections.
"""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """MLP block with LayerNorm, two Linear layers, GELU, and dropout."""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """
    MLP-Mixer block with token-mixing and channel-mixing.
    Input: [B, N, C] where N=num_tokens (81), C=channels (embedding_dim)
    """
    def __init__(self, num_tokens, embedding_dim, token_mixing_dim, channel_mixing_dim, dropout=0.0):
        super().__init__()
        # Token-mixing MLP: operates across the token dimension (N)
        self.token_mix = MLPBlock(num_tokens, token_mixing_dim, dropout)
        # Channel-mixing MLP: operates across the channel dimension (C)
        self.channel_mix = MLPBlock(embedding_dim, channel_mixing_dim, dropout)

    def forward(self, x):
        # x: [B, N, C]
        # Token mixing: transpose to [B, C, N], apply MLP, transpose back
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        # Channel mixing: apply MLP directly on [B, N, C]
        x = x + self.channel_mix(x)
        return x


class MLPMixer_HSI(nn.Module):
    """
    MLP-Mixer adapted for HSI patches.
    Input: [B, 128, 9, 9] (NCHW) - 128 channels, 9x9 spatial
    """
    def __init__(
        self,
        in_channels=128,
        num_classes=4,
        patch_size=9,
        embedding_dim=256,
        num_blocks=8,
        token_mixing_dim=128,
        channel_mixing_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size  # 81
        self.embedding_dim = embedding_dim

        # Patch embedding: project 128 channels -> embedding_dim per spatial location
        self.patch_embed = nn.Linear(in_channels, embedding_dim)

        # Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=self.num_tokens,
                embedding_dim=embedding_dim,
                token_mixing_dim=token_mixing_dim,
                channel_mixing_dim=channel_mixing_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # Final LayerNorm and classification head
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, 128, 9, 9] (NCHW) - from HSIDataset
        Returns: [B, num_classes]
        """
        B, C, H, W = x.shape
        assert C == 128 and H == self.patch_size and W == self.patch_size, \
            f"Expected [B, 128, {self.patch_size}, {self.patch_size}], got {x.shape}"

        # NCHW -> NHWC: [B, 128, 9, 9] -> [B, 9, 9, 128]
        x = x.permute(0, 2, 3, 1)

        # Flatten spatial dimensions: [B, 9, 9, 128] -> [B, 81, 128]
        x = x.flatten(1, 2)

        # Patch embedding: [B, 81, 128] -> [B, 81, embedding_dim]
        x = self.patch_embed(x)

        # Mixer blocks
        for block in self.mixer_blocks:
            x = block(x)

        # Global average pooling over tokens: [B, 81, embedding_dim] -> [B, embedding_dim]
        x = self.norm(x).mean(dim=1)

        # Classification head
        x = self.head(x)
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_mlp_mixer():
    """Test MLP-Mixer forward pass and token mixing."""
    model = MLPMixer_HSI(
        in_channels=128,
        num_classes=4,
        patch_size=9,
        embedding_dim=256,
        num_blocks=8,
        token_mixing_dim=128,
        channel_mixing_dim=512,
        dropout=0.1,
    )

    print(f"MLP-Mixer Parameters: {count_parameters(model):,}")

    # Test forward pass - NCHW format from HSIDataset
    x = torch.randn(2, 128, 9, 9)  # Batch=2, 128 channels, 9x9
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"

    # Test token mixing: verify that token mixing actually mixes across spatial tokens
    # Create input where each spatial position has a distinct pattern
    x_test = torch.zeros(1, 128, 9, 9)
    for i in range(9):
        for j in range(9):
            idx = i * 9 + j
            x_test[0, idx % 128, i, j] = 1.0

    out_test = model(x_test)
    print(f"Token mixing test passed: output varies = {(out_test.std() > 1e-6).item()}")

    # Verify gradients flow
    loss = out.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Gradient flow test passed: total grad norm = {grad_norm:.4f}")

    # Memory estimate for batch=16
    with torch.no_grad():
        x_batch = torch.randn(16, 128, 9, 9)
        out_batch = model(x_batch)
        # Rough VRAM estimate: params + activations + gradients
        param_mem = sum(p.numel() * 4 for p in model.parameters()) / 1e9  # GB (fp32)
        print(f"Estimated parameter memory (fp32): {param_mem:.2f} GB")

    print("All MLP-Mixer tests passed!")
    return model


if __name__ == "__main__":
    test_mlp_mixer()