"""
Isolation Forest model for anomaly detection.

Isolation Forest is used here as a simple unsupervised model that gives higher
scores to traffic rows that look unusual.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import config


class IsolationForestDetector:
    """Isolation Forest-based anomaly detector for network traffic."""

    def __init__(self, params=None):
        self.params = params or config.ISOLATION_FOREST_PARAMS
        self.model = None
        self.score_scaler = MinMaxScaler()  # For normalizing scores
        self.threshold = None

    def build_model(self):
        """Build the Isolation Forest model."""
        self.model = IsolationForest(
            n_estimators=self.params['n_estimators'],
            contamination=self.params['contamination'],
            max_samples=self.params['max_samples'],
            max_features=self.params['max_features'],
            random_state=self.params['random_state'],
            n_jobs=-1
        )
        return self.model

    def train(self, X_train):
        """Train the Isolation Forest model."""
        if self.model is None:
            self.build_model()

        print("Training Isolation Forest...")
        self.model.fit(X_train)

        # Get anomaly scores for training data to normalize
        raw_scores = self.model.decision_function(X_train)
        # Reshape for scaler
        raw_scores = raw_scores.reshape(-1, 1)
        self.score_scaler.fit(raw_scores)

        print("Isolation Forest training complete.")
        return self.model

    def predict_anomaly_scores(self, X):
        """
        Return normalized anomaly scores.
        Higher score = more anomalous (0 to 1 range)
        """
        # decision_function returns negative for anomalies, positive for normal
        raw_scores = self.model.decision_function(X)

        # Normalize to 0-1 range
        normalized_scores = self.score_scaler.transform(raw_scores.reshape(-1, 1)).flatten()

        # Invert so higher = more anomalous
        anomaly_scores = 1 - normalized_scores

        return anomaly_scores

    def predict_raw_scores(self, X):
        """Return raw decision function scores."""
        return self.model.decision_function(X)

    def set_threshold(self, X_train, percentile=None):
        """Set the anomaly threshold using a percentile of training scores."""
        percentile = percentile or config.ENSEMBLE_PARAMS['anomaly_percentile']
        scores = self.predict_anomaly_scores(X_train)
        self.threshold = np.percentile(scores, percentile)

        print(f"Isolation Forest threshold set to {self.threshold:.6f}")
        print(f"  Method: {percentile}th percentile of anomaly scores")
        print(f"  Stats: min={scores.min():.6f}, mean={scores.mean():.6f}, max={scores.max():.6f}")
        print(f"  Expected FPR on normal data: ~{100-percentile:.0f}%")

        return self.threshold

    def predict(self, X, threshold=None):
        """
        Predict anomalies.
        Returns 1 for anomaly, 0 for normal.
        """
        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            # Use model's predict function as fallback
            predictions = self.model.predict(X)
            return (predictions == -1).astype(int)

        scores = self.predict_anomaly_scores(X)
        return (scores > threshold).astype(int)

    def get_feature_importance(self, X):
        """
        Estimate feature importance based on tree paths.
        This is an approximation for interpretability.
        """
        # Isolation Forest doesn't have built-in feature importance,
        # but we can estimate it from the number of splits
        n_features = X.shape[1]
        feature_counts = np.zeros(n_features)

        for estimator in self.model.estimators_:
            tree = estimator.tree_
            feature = tree.feature
            counts = np.bincount(feature[feature >= 0], minlength=n_features)
            feature_counts += counts

        # Normalize
        if feature_counts.sum() > 0:
            feature_importance = feature_counts / feature_counts.sum()
        else:
            feature_importance = np.ones(n_features) / n_features

        return feature_importance

    def save_model(self, path=None):
        """Save the trained model."""
        path = path or config.MODEL_DIR
        with open(os.path.join(path, 'isolation_forest.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(path, 'isolation_forest_scaler.pkl'), 'wb') as f:
            pickle.dump(self.score_scaler, f)
        if self.threshold is not None:
            np.save(os.path.join(path, 'isolation_forest_threshold.npy'), np.array([self.threshold]))
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load a trained model."""
        path = path or config.MODEL_DIR
        with open(os.path.join(path, 'isolation_forest.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(path, 'isolation_forest_scaler.pkl'), 'rb') as f:
            self.score_scaler = pickle.load(f)
        threshold_path = os.path.join(path, 'isolation_forest_threshold.npy')
        if os.path.exists(threshold_path):
            self.threshold = np.load(threshold_path)[0]
        print(f"Model loaded from {path}")


def train_isolation_forest(X_train, contamination=None):
    """
    Train Isolation Forest on the training data.
    """
    params = config.ISOLATION_FOREST_PARAMS.copy()
    if contamination is not None:
        params['contamination'] = contamination

    detector = IsolationForestDetector(params)
    detector.train(X_train)
    detector.set_threshold(X_train)

    return detector


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()

    detector = train_isolation_forest(data['X_train'])

    # Test predictions
    scores = detector.predict_anomaly_scores(data['X_test'][:10])
    predictions = detector.predict(data['X_test'][:10])
    print(f"\nSample anomaly scores: {scores[:5]}")
    print(f"Sample predictions: {predictions[:5]}")
