import json
import numpy as np
from sklearn.metrics import confusion_matrix

def main():
    with open('artifacts/dmlpffn_test_logits_frozen.json', 'r') as f:
        data = json.load(f)

    logits = np.array(data['logits'])
    y_true = np.array(data['true_labels'])

    print(f"Total samples: {len(y_true)}")
    unique, counts = np.unique(y_true, return_counts=True)
    print(f"True class counts: {dict(zip(unique, counts))}")

    # 4-class predictions from argmax
    y_pred_4 = np.argmax(logits, axis=1)
    cm_4 = confusion_matrix(y_true, y_pred_4, labels=[0, 1, 2, 3])
    print("\n4-class Confusion Matrix:")
    print(cm_4)

    acc_4 = np.sum(np.diag(cm_4)) / len(y_true)
    print(f"4-class Accuracy: {acc_4:.4f}")

    # Conditional Binary: Healthy (0) vs Late (3)
    # P(H|H/L) = exp(z0) / (exp(z0) + exp(z3))
    # Prediction: 0 if z0 > z3 else 3 (Late)
    z0 = logits[:, 0]
    z3 = logits[:, 3]

    # Map y_true: 0 -> 0, 3 -> 1, others excluded from conditional binary
    mask = (y_true == 0) | (y_true == 3)
    y_true_subset = y_true[mask]
    z0_subset = z0[mask]
    z3_subset = z3[mask]

    y_true_binary = np.zeros(len(y_true_subset), dtype=int)
    y_true_binary[y_true_subset == 3] = 1

    y_pred_binary = (z3_subset > z0_subset).astype(int)

    cm_bin = confusion_matrix(y_true_binary, y_pred_binary)
    print("\nConditional Binary Confusion Matrix (from logits, z0 vs z3):")
    print(cm_bin)

    tn, fp, fn, tp = cm_bin.ravel()
    acc_bin = (tn + tp) / len(y_true_binary)
    print(f"Conditional Binary Accuracy: {acc_bin:.4f}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    print("\n--- From artifact file ---")
    print("Conditional binary metrics:", data['conditional_binary_metrics'])
    print("Confusion matrix (logits):", data['confusion_matrix_conditional_binary_logits'])

if __name__ == "__main__":
    main()
