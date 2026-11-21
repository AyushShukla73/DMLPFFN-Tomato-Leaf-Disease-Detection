import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from config import cfg
from dmlpffn_disease import DMLPFFN
from data_preprocessing import HSIDataset

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce)
        loss = self.alpha * (1-pt)**self.gamma * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()


def train_model(dataloaders, in_ch, num_classes, optimizer_type='Adam', loss_type='CrossEntropyLoss', lr=5e-4, weight_decay=1e-4, epochs=500, patience=20, scheduler_type='ReduceLROnPlateau', scheduler_params=None):
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
        sched = None  # No scheduler if not specified

    # Compute class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    y_train = []
    for xb, yb in dataloaders['train']:
        y_train.extend(yb.numpy())
    y_train = np.array(y_train)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Add weighted sampling for balanced batches
    from torch.utils.data import WeightedRandomSampler
    class_sample_counts = np.bincount(y_train)
    sample_weights = 1. / class_sample_counts[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    dataloaders['train'] = torch.utils.data.DataLoader(
        dataloaders['train'].dataset,
        batch_size=dataloaders['train'].batch_size,
        sampler=sampler,
        num_workers=dataloaders['train'].num_workers
    )

    # Select loss function
    if loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'FocalLoss':
        criterion = FocalLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

    best_acc, epochs_no_improve = 0.0, 0
    best_state = None
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in dataloaders['train']:
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

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if sched is not None:
            sched.step(val_acc)
        if val_acc > best_acc:
            best_acc, epochs_no_improve = val_acc, 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
