import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader, device=None):
    """Evaluate model on a dataloader."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu()
            y_pred.append(preds)
            y_true.append(yb.cpu())
    y_true = np.concatenate([y.numpy() for y in y_true])
    y_pred = np.concatenate([y.numpy() for y in y_pred])
    report = classification_report(y_true, y_pred, target_names=['Healthy', 'Asymptomatic', 'Early', 'Late'], digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm