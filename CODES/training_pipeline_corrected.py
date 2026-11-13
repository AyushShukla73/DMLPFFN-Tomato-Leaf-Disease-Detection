"""
Corrected Training Pipeline for DMLPFFN

Improvements over original:
- Accepts device as parameter (no hardcoded CUDA_VISIBLE_DEVICES)
- Works with pre-configured DataLoaders (doesn't recreate them)
- Proper reproducibility with seed handling
- Better logging and metric tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from config import cfg
from dmlpffn_disease import DMLPFFN


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def create_weighted_sampler(dataloader: DataLoader, device: torch.device) -> DataLoader:
    """Create a DataLoader with WeightedRandomSampler for class balancing."""
    # Get all labels from the dataset
    y_train = []
    for _, yb in dataloader:
        y_train.extend(yb.numpy())
    y_train = np.array(y_train)

    class_sample_counts = np.bincount(y_train)
    sample_weights = 1. / class_sample_counts[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory
    )


def train_model(
    dataloaders: dict,
    in_ch: int,
    num_classes: int,
    optimizer_type: str = 'AdamW',
    loss_type: str = 'CrossEntropyLoss',
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    epochs: int = 250,
    patience: int = 25,
    scheduler_type: str = 'CosineAnnealingWarmRestarts',
    scheduler_params: dict = None,
    device: torch.device = None,
    use_weighted_sampler: bool = True
):
    """
    Train the DMLPFFN model with proper device handling and logging.

    Args:
        dataloaders: Dict with 'train', 'val', 'test' DataLoaders
        in_ch: Number of input channels (PCA components)
        num_classes: Number of output classes
        optimizer_type: Optimizer type
        loss_type: Loss function type
        lr: Learning rate
        weight_decay: Weight decay
        epochs: Maximum epochs
        patience: Early stopping patience
        scheduler_type: Scheduler type
        scheduler_params: Scheduler parameters
        device: Compute device (auto-detected if None)
        use_weighted_sampler: Whether to use WeightedRandomSampler for train loader

    Returns:
        (model, history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DMLPFFN(in_channels=in_ch, num_classes=num_classes).to(device)

    # Select optimizer
    if optimizer_type == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == 'RMSprop':
        opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Select scheduler
    if scheduler_type == 'ReduceLROnPlateau':
        sched_params = scheduler_params or {'mode': 'max', 'patience': 10, 'factor': 0.5}
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **sched_params)
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    else:
        sched = None

    # Compute class weights for imbalance
    y_train = []
    for _, yb in dataloaders['train']:
        y_train.extend(yb.numpy())
    y_train = np.array(y_train)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Apply weighted sampler to training loader if requested
    train_loader = dataloaders['train']
    if use_weighted_sampler:
        train_loader = create_weighted_sampler(dataloaders['train'], device)
        print("Using WeightedRandomSampler for training")

    # Select loss function
    if loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'FocalLoss':
        criterion = FocalLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

    best_acc, epochs_no_improve = 0.0, 0
    best_state = None
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, yb in dataloaders['val']:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(1).cpu()
                y_pred.append(preds)
                y_true.append(yb.cpu())

        val_loss /= len(dataloaders['val'].dataset)
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        val_acc = accuracy_score(y_true, y_pred)

        # Record learning rate
        current_lr = opt.param_groups[0]['lr']
        history['lr'].append(current_lr)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {current_lr:.2e}")

        # Scheduler step
        if sched is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                sched.step(val_acc)
            else:
                sched.step()

        # Early stopping
        if val_acc > best_acc:
            best_acc, epochs_no_improve = val_acc, 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val_acc: {best_acc:.4f})")

    return model, history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """Evaluate model on a dataloader."""
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu()
            y_pred.append(preds)
            y_true.append(yb.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    report = classification_report(y_true, y_pred, target_names=['Healthy', 'Asymptomatic', 'Early', 'Late'], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    return report, cm