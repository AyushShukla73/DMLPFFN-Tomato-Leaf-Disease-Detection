import torch
from torch.utils.data import DataLoader
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from config import cfg
from data_preprocessing import load_tomato_dataset, spectral_normalize, apply_pca, train_val_test_split, HSIDataset
from training_pipeline import train_model
from evaluation_metrics import evaluate_model

if __name__ == '__main__':
    # 1) Load data with patch extraction and per-image PCA
    X, y = load_tomato_dataset(cfg.data_root, patch_size=cfg.patch_size, stride=cfg.stride)
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")

    # 2) Spectral normalization
    X = spectral_normalize(X)
    print(f"After normalization: X shape {X.shape}")

    # Best hyperparameters config (temporarily reduced for small sample test)
    config = {
        "optimizer": "AdamW",
        "loss_function": "CrossEntropyLoss",
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "epochs": 200,
        "patience": cfg.patience,
        "scheduler": "CosineAnnealingWarmRestarts",
        "scheduler_params": None
    }

    print("Using best hyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train final model with best params on full data
    splits = train_val_test_split(X, y)
    dls = {k: DataLoader(HSIDataset(*v, augment=(k=='train')), batch_size=config['batch_size'], shuffle=(k=='train')) for k,v in splits.items()}
    input_dim = X.shape[3]  # B after PCA (channels)
    model, history = train_model(dls, in_ch=input_dim, num_classes=cfg.num_classes,
                        optimizer_type=config['optimizer'], loss_type=config['loss_function'],
                        lr=config['lr'], weight_decay=config['weight_decay'],
                        epochs=config['epochs'], patience=config['patience'],
                        scheduler_type=config['scheduler'], scheduler_params=config['scheduler_params'])

    # Plot training history
    import matplotlib.pyplot as plt
    epochs_range = range(1, len(history['train_acc']) + 1)

    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dls['test']:
            xb = xb.to(model.parameters().__iter__().__next__().device)
            logits = model(xb)
            preds = logits.argmax(1).cpu()
            y_pred.append(preds)
            y_true.append(yb)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Asymptomatic', 'Early', 'Late'])
    disp.plot(ax=plt.gca(), cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_plot.png')
    plt.show()

    # Evaluate on test set
    test_loader = dls['test']
    report, cm = evaluate_model(model, test_loader)
    print("Final test results:")
    print(report)
    print(cm)
