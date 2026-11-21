"""
Unified Trainer for Deep Learning Baselines
Uses Phase 3 training configuration: mixup, cutmix, label smoothing, weighted sampler, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """CutMix augmentation for 9x9 patches in NCHW format."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    # x is in NCHW format: [B, C=128, H=9, W=9]
    B, C, H, W = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

    # Random center for the cut region
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Ensure valid bounding box (clipping to [0, W/H])
    bbx1 = max(0, int(cx - cut_w // 2))
    bby1 = max(0, int(cy - cut_h // 2))
    bbx2 = min(W, int(cx + cut_w // 2))
    bby2 = min(H, int(cy + cut_h // 2))

    mixed_x = x.clone()
    # Only apply cutmix if the bounding box is valid
    if bbx2 > bbx1 and bby2 > bby1:
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to match actual cut ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss computation for mixed samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_class_weights(y, num_classes=4):
    """Compute class weights for WeightedRandomSampler."""
    class_counts = np.bincount(y, minlength=num_classes)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    sample_weights = class_weights[y]
    return sample_weights, class_weights


class BaselineTrainer:
    """
    Unified trainer for all deep learning baselines.
    Uses Phase 3 configuration exactly.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        output_dir: str = 'artifacts/baselines',
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Phase 3 Configuration (fixed for all baselines)
        # Support both dict and TrainingConfig dataclass
        def get_config(key, default):
            if hasattr(config, 'get'):
                return config.get(key, default)
            return getattr(config, key, default)

        self.epochs = get_config('epochs', 250)
        self.lr = get_config('lr', 2e-4)
        self.weight_decay = get_config('weight_decay', 5e-4)
        self.label_smoothing = get_config('label_smoothing', 0.1)
        self.mixup_alpha = get_config('mixup_alpha', 0.2)
        self.cutmix_alpha = get_config('cutmix_alpha', 1.0)
        self.mixup_prob = get_config('mixup_prob', 0.5)
        self.cutmix_prob = get_config('cutmix_prob', 0.5)
        self.patience = get_config('patience', 25)
        self.scheduler_T0 = get_config('scheduler_T0', 10)
        self.use_focal_loss = get_config('use_focal_loss', False)
        self.focal_alpha = get_config('focal_alpha', 0.25)
        self.focal_gamma = get_config('focal_gamma', 2.0)

        # Loss function
        if self.use_focal_loss:
            self.criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.scheduler_T0
        )

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # Apply augmentation with probability
            use_mixup = np.random.random() < self.mixup_prob and self.mixup_alpha > 0
            use_cutmix = np.random.random() < self.cutmix_prob and self.cutmix_alpha > 0

            if use_mixup and use_cutmix:
                # Choose one randomly
                if np.random.random() < 0.5:
                    use_cutmix = False
                else:
                    use_mixup = False

            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha, self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
            elif use_cutmix:
                x, y_a, y_b, lam = cutmix_data(x, y, self.cutmix_alpha, self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
            else:
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def validate(self, loader: DataLoader, save_logits: bool = False, save_groups: bool = False, output_dir: Path = None) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Validate on given loader. Optionally save logits and group info."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_logits = [] if save_logits else None

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                if save_logits:
                    all_logits.extend(outputs.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total

        # Save logits if requested
        if save_logits and output_dir is not None:
            np.save(output_dir / "test_logits.npy", np.array(all_logits))

        # Save groups if requested
        if save_groups and output_dir is not None:
            # Extract groups from dataset
            if hasattr(loader.dataset, 'groups'):
                groups = loader.dataset.groups
            elif hasattr(loader.dataset, 'indices'):
                # For Subset datasets
                parent_dataset = loader.dataset.dataset
                if hasattr(parent_dataset, 'groups'):
                    groups = [parent_dataset.groups[i] for i in loader.dataset.indices]
                else:
                    groups = None
            else:
                groups = None

            if groups is not None:
                np.save(output_dir / "test_groups.npy", np.array(groups))

        return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

    def train(self, **kwargs) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        print(f"Starting training for {self.epochs} epochs (patience={self.patience})")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, _, _ = self.validate(self.val_loader)

            # Scheduler step
            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start

            # Print progress
            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                # Save best model
                self.save_checkpoint('best_model.pth')
                print(f"  -> New best val acc: {val_acc:.4f}")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")

        # Load best model for final evaluation
        self.load_checkpoint('best_model.pth')

        # Final test evaluation - save logits and groups for statistical analysis
        test_loss, test_acc, test_preds, test_targets = self.validate(
            self.test_loader,
            save_logits=True,
            save_groups=True,
            output_dir=self.output_dir
        )
        print(f"Test Accuracy: {test_acc:.4f}")

        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'train_time': total_time,
            'history': self.history,
            'test_predictions': test_preds,
            'test_targets': test_targets,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        # Convert config to dict for safe serialization
        if hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
        elif hasattr(self.config, 'get'):
            config_dict = dict(self.config)
        else:
            config_dict = {}

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': config_dict,
            'seed': self.seed,
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def create_weighted_sampler(dataset, num_classes=4):
    """Create WeightedRandomSampler for class balancing."""
    # Get all labels from dataset
    labels = []
    for _, y in dataset:
        labels.append(y.item() if isinstance(y, torch.Tensor) else y)
    labels = np.array(labels)

    sample_weights, _ = get_class_weights(labels, num_classes)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_phase3_config() -> Dict[str, Any]:
    """Return Phase 3 training configuration (fixed for all baselines)."""
    return {
        'epochs': 250,
        'lr': 2e-4,
        'weight_decay': 5e-4,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'mixup_prob': 0.5,
        'cutmix_prob': 0.5,
        'patience': 25,
        'scheduler_T0': 10,
        'use_focal_loss': False,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'batch_size': 16,
    }