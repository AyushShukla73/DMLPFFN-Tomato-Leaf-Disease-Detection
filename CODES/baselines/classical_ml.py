"""
Classical ML Baselines for Hyperspectral Imaging
SVM (RBF) and Random Forest on flattened HSI features.
"""

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import torch


class ClassicalMLBaselines:
    """
    Classical ML baselines for HSI patch classification.
    All models use flattened features: [N, 9*9*128] = [N, 10368]
    Optional: PCA to 256 components before classifier.
    """

    @staticmethod
    def get_svm_pipeline(pca_components=256, C=10.0, gamma='scale', random_state=42):
        """
        SVM with RBF kernel and optional PCA.
        """
        steps = []
        if pca_components and pca_components < 10368:
            steps.append(('pca', PCA(n_components=pca_components, random_state=random_state)))
        steps.append(('scaler', StandardScaler()))
        steps.append(('svc', SVC(
            kernel='rbf',
            C=C,
            gamma=gamma,
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )))
        return Pipeline(steps)

    @staticmethod
    def get_rf_pipeline(pca_components=256, n_estimators=500, max_depth=20, random_state=42):
        """
        Random Forest with optional PCA.
        """
        steps = []
        if pca_components and pca_components < 10368:
            steps.append(('pca', PCA(n_components=pca_components, random_state=random_state)))
        steps.append(('scaler', StandardScaler()))
        steps.append(('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )))
        return Pipeline(steps)


def prepare_classical_ml_data(X, y):
    """
    Prepare data for classical ML.
    X: [N, 9, 9, 128] (NHWC) -> flatten to [N, 10368]
    y: [N] integer labels
    """
    N = X.shape[0]
    X_flat = X.reshape(N, -1)  # [N, 9*9*128]
    return X_flat, y


def train_svm(X_train, y_train, X_val=None, y_val=None, pca_components=256):
    """Train SVM pipeline and optionally evaluate on validation."""
    pipeline = ClassicalMLBaselines.get_svm_pipeline(pca_components=pca_components)
    pipeline.fit(X_train, y_train)

    results = {'model': pipeline}
    if X_val is not None and y_val is not None:
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)
        results['val_predictions'] = y_pred
        results['val_probabilities'] = y_proba
        results['val_accuracy'] = (y_pred == y_val).mean()

    return results


def train_rf(X_train, y_train, X_val=None, y_val=None, pca_components=256):
    """Train Random Forest pipeline and optionally evaluate on validation."""
    pipeline = ClassicalMLBaselines.get_rf_pipeline(pca_components=pca_components)
    pipeline.fit(X_train, y_train)

    results = {'model': pipeline}
    if X_val is not None and y_val is not None:
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)
        results['val_predictions'] = y_pred
        results['val_probabilities'] = y_proba
        results['val_accuracy'] = (y_pred == y_val).mean()

    return results


def save_classical_model(model, path):
    """Save classical ML model to disk."""
    joblib.dump(model, path)


def load_classical_model(path):
    """Load classical ML model from disk."""
    return joblib.load(path)


def test_classical_ml():
    """Test classical ML pipelines with dummy data."""
    print("=== Testing Classical ML Pipelines ===")

    # Create dummy data
    N_train = 1000
    N_val = 200
    X_train = np.random.randn(N_train, 9, 9, 128).astype(np.float32)
    y_train = np.random.randint(0, 4, N_train)
    X_val = np.random.randn(N_val, 9, 9, 128).astype(np.float32)
    y_val = np.random.randint(0, 4, N_val)

    # Prepare data
    X_train_flat, y_train_flat = prepare_classical_ml_data(X_train, y_train)
    X_val_flat, y_val_flat = prepare_classical_ml_data(X_val, y_val)

    print(f"Train shape: {X_train_flat.shape}")
    print(f"Val shape: {X_val_flat.shape}")

    # Test SVM
    print("\n--- Testing SVM ---")
    svm_results = train_svm(X_train_flat, y_train_flat, X_val_flat, y_val_flat, pca_components=256)
    print(f"SVM Val Accuracy: {svm_results['val_accuracy']:.4f}")

    # Test RF
    print("\n--- Testing Random Forest ---")
    rf_results = train_rf(X_train_flat, y_train_flat, X_val_flat, y_val_flat, pca_components=256)
    print(f"RF Val Accuracy: {rf_results['val_accuracy']:.4f}")

    # Test save/load
    print("\n--- Testing Save/Load ---")
    save_classical_model(svm_results['model'], 'test_svm.joblib')
    loaded_svm = load_classical_model('test_svm.joblib')
    print(f"Loaded SVM type: {type(loaded_svm)}")

    save_classical_model(rf_results['model'], 'test_rf.joblib')
    loaded_rf = load_classical_model('test_rf.joblib')
    print(f"Loaded RF type: {type(loaded_rf)}")

    print("All Classical ML tests passed!")
    return svm_results, rf_results


if __name__ == "__main__":
    test_classical_ml()