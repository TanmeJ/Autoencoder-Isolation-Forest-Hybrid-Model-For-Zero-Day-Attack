"""
Hybrid anomaly detector.

This file combines an Autoencoder score and an Isolation Forest score by taking
a weighted average. It is kept simple so the training flow is easy to follow.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import os
import config
from autoencoder_model import AutoencoderAnomalyDetector, train_autoencoder_on_normal_data
from isolation_forest_model import IsolationForestDetector, train_isolation_forest


class EnsembleAnomalyDetector:
    """Hybrid detector combining Autoencoder and Isolation Forest."""

    def __init__(self, autoencoder_weight=None, isolation_forest_weight=None):
        self.autoencoder_weight = autoencoder_weight or config.ENSEMBLE_PARAMS['autoencoder_weight']
        self.isolation_forest_weight = isolation_forest_weight or config.ENSEMBLE_PARAMS['isolation_forest_weight']
        self.autoencoder = None
        self.isolation_forest = None
        self.threshold = None

        # Normalize weights
        total = self.autoencoder_weight + self.isolation_forest_weight
        self.autoencoder_weight /= total
        self.isolation_forest_weight /= total

    def train(self, X_train, y_train, normal_class, X_val=None):
        """Train both models and set thresholds."""
        print("=" * 60)
        print("Training Ensemble Anomaly Detector")
        print("=" * 60)

        # Train Autoencoder on normal data only
        print("\n[1/2] Training Autoencoder...")
        self.autoencoder = train_autoencoder_on_normal_data(
            X_train, y_train, normal_class, X_val
        )

        # Train Isolation Forest on all data
        print("\n[2/2] Training Isolation Forest...")
        self.isolation_forest = train_isolation_forest(X_train)

        # Set ensemble threshold using normal data
        X_normal = X_train[y_train == normal_class]
        self.set_threshold(X_normal)

        print("\n" + "=" * 60)
        print("Ensemble Training Complete!")
        print("=" * 60)

    def set_threshold(self, X_normal, percentile=None):
        """Set threshold using a percentile of normal training scores."""
        percentile = percentile or config.ENSEMBLE_PARAMS['anomaly_percentile']
        scores = self.predict_anomaly_scores(X_normal)
        self.threshold = np.percentile(scores, percentile)
        print(f"Ensemble threshold set to {self.threshold:.6f} (percentile {percentile})")
        print(f"  Interpretation: Scores above {self.threshold:.6f} are flagged as attacks")
        print(f"  Expected FPR on normal data: ~{100-percentile:.0f}%")

    def calibrate_threshold_f1(self, X_val, y_val, normal_class):
        """
        Find optimal threshold using F1 score on validation data.

        This method searches for the threshold that maximizes F1 score
        on labeled validation data. This is useful when you have labeled
        validation data and want to optimize for F1 score.

        Args:
            X_val: Validation features
            y_val: Validation labels (encoded)
            normal_class: Encoded label for normal class

        Returns:
            optimal_threshold: Threshold that maximizes F1
            best_f1: Best F1 score achieved
        """
        y_binary = (y_val != normal_class).astype(int)
        scores = self.predict_anomaly_scores(X_val)

        # Search for optimal threshold
        thresholds = np.percentile(scores, np.arange(50, 100, 1))
        best_f1 = 0
        best_threshold = self.threshold

        for thresh in thresholds:
            preds = (scores > thresh).astype(int)
            f1 = f1_score(y_binary, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        print(f"\nThreshold Calibration (F1-optimal):")
        print(f"  Previous threshold: {self.threshold:.6f}")
        print(f"  Optimal threshold:  {best_threshold:.6f}")
        print(f"  Best F1 score:      {best_f1:.4f}")

        self.threshold = best_threshold
        return best_threshold, best_f1

    def predict_anomaly_scores(self, X):
        """
        Combine scores from both models.
        Returns weighted average of normalized anomaly scores.
        """
        # Get scores from each model
        ae_scores = self.autoencoder.predict_anomaly_scores(X)
        if_scores = self.isolation_forest.predict_anomaly_scores(X)

        # Normalize scores to [0, 1] range
        ae_scores_norm = self._normalize_scores(ae_scores)
        if_scores_norm = self._normalize_scores(if_scores)

        # Weighted combination
        ensemble_scores = (
            self.autoencoder_weight * ae_scores_norm +
            self.isolation_forest_weight * if_scores_norm
        )

        return ensemble_scores

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] range."""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val > 0:
            return (scores - min_val) / (max_val - min_val)
        return scores

    def predict(self, X, threshold=None):
        """
        Predict anomalies.
        Returns 1 for anomaly (attack), 0 for normal.
        """
        if threshold is None:
            threshold = self.threshold

        scores = self.predict_anomaly_scores(X)
        return (scores > threshold).astype(int)

    def get_individual_scores(self, X):
        """Get scores from each model separately."""
        return {
            'autoencoder': self.autoencoder.predict_anomaly_scores(X),
            'isolation_forest': self.isolation_forest.predict_anomaly_scores(X),
            'ensemble': self.predict_anomaly_scores(X)
        }

    def evaluate(self, X_test, y_test, normal_class):
        """Evaluate the hybrid model and both individual models."""
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)

        # Convert labels: normal=0, attack=1
        y_binary = (y_test != normal_class).astype(int)

        # Get predictions
        ensemble_pred = self.predict(X_test)
        ae_pred = self.autoencoder.predict(X_test)
        if_pred = self.isolation_forest.predict(X_test)

        # Get scores for ROC-AUC
        ensemble_scores = self.predict_anomaly_scores(X_test)
        ae_scores = self.autoencoder.predict_anomaly_scores(X_test)
        if_scores = self.isolation_forest.predict_anomaly_scores(X_test)

        results = {}

        for name, pred, scores in [
            ('Autoencoder', ae_pred, ae_scores),
            ('Isolation Forest', if_pred, if_scores),
            ('Ensemble', ensemble_pred, ensemble_scores)
        ]:
            print(f"\n{name} Performance:")
            print("-" * 40)

            acc = accuracy_score(y_binary, pred)
            prec = precision_score(y_binary, pred, zero_division=0)
            rec = recall_score(y_binary, pred, zero_division=0)
            f1 = f1_score(y_binary, pred, zero_division=0)

            try:
                auc = roc_auc_score(y_binary, scores)
            except ValueError:
                auc = 0.0

            # Confusion matrix
            cm = confusion_matrix(y_binary, pred)
            tn, fp, fn, tp = cm.ravel()

            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC-ROC:   {auc:.4f}")
            print(f"FPR:       {fpr:.4f}")

            print(f"\nConfusion Matrix:")
            print(f"  TN: {tn}, FP: {fp}")
            print(f"  FN: {fn}, TP: {tp}")

            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'fpr': fpr,
                'predictions': pred,
                'scores': scores,
                'confusion_matrix': cm
            }

        # Print comparison table
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'FPR':>8}")
        print("-" * 68)
        for name in ['Autoencoder', 'Isolation Forest', 'Ensemble']:
            if name in results:
                r = results[name]
                print(f"{name:<20} {r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['fpr']:>8.4f}")
        print("=" * 68)

        return results

    def plot_roc_curves(self, X_test, y_test, normal_class, save_path=None):
        """Plot ROC curves for all models."""
        y_binary = (y_test != normal_class).astype(int)

        ae_scores = self.autoencoder.predict_anomaly_scores(X_test)
        if_scores = self.isolation_forest.predict_anomaly_scores(X_test)
        ensemble_scores = self.predict_anomaly_scores(X_test)

        plt.figure(figsize=(10, 8))

        for name, scores, color in [
            ('Autoencoder', ae_scores, 'blue'),
            ('Isolation Forest', if_scores, 'green'),
            ('Ensemble', ensemble_scores, 'red')
        ]:
            fpr, tpr, _ = roc_curve(y_binary, scores)
            auc = roc_auc_score(y_binary, scores)
            plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Network Anomaly Detection')
        plt.legend(loc='lower right')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.close()

    def save_models(self, path=None):
        """Save all models."""
        path = path or config.MODEL_DIR
        self.autoencoder.save_model(path)
        self.isolation_forest.save_model(path)
        if self.threshold is not None:
            np.save(os.path.join(path, 'ensemble_threshold.npy'), np.array([self.threshold]))
        print(f"All models saved to {path}")

    def load_models(self, path=None):
        """Load all models."""
        path = path or config.MODEL_DIR
        self.autoencoder = AutoencoderAnomalyDetector(input_dim=0)
        self.autoencoder.load_model(path)
        self.isolation_forest = IsolationForestDetector()
        self.isolation_forest.load_model(path)
        threshold_path = os.path.join(path, 'ensemble_threshold.npy')
        if os.path.exists(threshold_path):
            self.threshold = np.load(threshold_path)[0]
        print(f"All models loaded from {path}")


def run_ensemble_detection(X_train, y_train, X_test, y_test, normal_class, X_val=None):
    """Complete pipeline for ensemble-based zero-day attack detection."""

    # Initialize ensemble
    ensemble = EnsembleAnomalyDetector()

    # Train
    ensemble.train(X_train, y_train, normal_class, X_val)

    # Evaluate
    results = ensemble.evaluate(X_test, y_test, normal_class)

    # Plot ROC curves
    save_path = os.path.join(config.RESULTS_DIR, 'roc_curves.png')
    ensemble.plot_roc_curves(X_test, y_test, normal_class, save_path)

    return ensemble, results


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    # Run complete pipeline
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()

    ensemble, results = run_ensemble_detection(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['normal_class'], data['X_val']
    )

    # Save models
    ensemble.save_models()
