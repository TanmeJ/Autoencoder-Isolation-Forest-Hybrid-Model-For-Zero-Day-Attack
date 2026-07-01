"""
Autoencoder model for anomaly detection.

The model is trained mostly on normal traffic. If a row is reconstructed badly,
the reconstruction error is treated as an anomaly score.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import os
import config


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detector for network traffic."""

    def __init__(self, input_dim, params=None):
        self.params = params or config.AUTOENCODER_PARAMS
        self.input_dim = input_dim
        self.model = None
        self.threshold = None
        self.history = None

    def build_model(self):
        """Build the autoencoder architecture."""
        encoding_dims = self.params['encoding_dims']
        latent_dim = self.params['latent_dim']
        activation = self.params['activation']
        output_activation = self.params['output_activation']

        # Input layer
        inputs = layers.Input(shape=(self.input_dim,))

        # Encoder
        x = inputs
        for dim in encoding_dims:
            x = layers.Dense(dim, activation=activation)(x)
            x = layers.BatchNormalization()(x)

        # Latent space
        encoded = layers.Dense(latent_dim, activation=activation, name='latent')(x)

        # Decoder
        for dim in reversed(encoding_dims):
            x = layers.Dense(dim, activation=activation)(x)
            x = layers.BatchNormalization()(x)

        # Output
        outputs = layers.Dense(self.input_dim, activation=output_activation)(x)

        # Build autoencoder
        autoencoder = models.Model(inputs, outputs, name='autoencoder')
        encoder = models.Model(inputs, encoded, name='encoder')

        self.model = autoencoder
        self.encoder = encoder

        return autoencoder

    def compile_model(self):
        """Compile the model with optimizer and loss."""
        learning_rate = self.params['learning_rate']
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def train(self, X_train, X_val=None):
        """Train the autoencoder on normal data."""
        if self.model is None:
            self.build_model()
            self.compile_model()

        print("Training Autoencoder...")

        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.params['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )

        # Model checkpoint
        checkpoint_path = os.path.join(config.MODEL_DIR, 'autoencoder_best.h5')
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )

        # Train
        validation_data = (X_val, X_val) if X_val is not None else None
        self.history = self.model.fit(
            X_train, X_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        return self.history

    def calculate_reconstruction_error(self, X):
        """Calculate reconstruction error for each sample."""
        predictions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        return mse

    def set_threshold(self, X_train, percentile=None):
        """Set the anomaly threshold using a percentile of training errors."""
        percentile = percentile or config.ENSEMBLE_PARAMS['anomaly_percentile']
        errors = self.calculate_reconstruction_error(X_train)
        self.threshold = np.percentile(errors, percentile)

        print(f"Autoencoder threshold set to {self.threshold:.6f}")
        print(f"  Method: {percentile}th percentile of normal reconstruction errors")
        print(f"  Stats: min={errors.min():.6f}, mean={errors.mean():.6f}, max={errors.max():.6f}")
        print(f"  Expected FPR on normal data: ~{100-percentile:.0f}%")

        return self.threshold

    def predict_anomaly_scores(self, X):
        """Return anomaly scores (reconstruction errors)."""
        return self.calculate_reconstruction_error(X)

    def predict(self, X, threshold=None):
        """Predict anomalies based on reconstruction error."""
        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        errors = self.predict_anomaly_scores(X)
        return (errors > threshold).astype(int)

    def save_model(self, path=None):
        """Save the trained model."""
        path = path or config.MODEL_DIR
        self.model.save(os.path.join(path, 'autoencoder.h5'))
        if self.threshold is not None:
            np.save(os.path.join(path, 'autoencoder_threshold.npy'), np.array([self.threshold]))
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load a trained model."""
        path = path or config.MODEL_DIR
        # Load without recompiling to avoid metric/loss deserialization issues
        # across different Keras/TensorFlow versions.
        self.model = keras.models.load_model(
            os.path.join(path, 'autoencoder.h5'),
            compile=False
        )
        threshold_path = os.path.join(path, 'autoencoder_threshold.npy')
        if os.path.exists(threshold_path):
            self.threshold = np.load(threshold_path)[0]
        print(f"Model loaded from {path}")


def train_autoencoder_on_normal_data(X_train, y_train, normal_class, X_val=None):
    """
    Train autoencoder only on normal data for better anomaly detection.
    """
    # Filter only normal samples
    normal_mask = y_train == normal_class
    X_normal = X_train[normal_mask]

    print(f"\nTraining autoencoder on {len(X_normal)} normal samples...")

    detector = AutoencoderAnomalyDetector(input_dim=X_train.shape[1])
    detector.train(X_normal, X_val if X_val is not None else X_normal)
    detector.set_threshold(X_normal)

    return detector


if __name__ == "__main__":
    # Quick test
    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()

    detector = train_autoencoder_on_normal_data(
        data['X_train'], data['y_train'], data['normal_class'], data['X_val']
    )

    # Test predictions
    scores = detector.predict_anomaly_scores(data['X_test'][:10])
    print(f"\nSample anomaly scores: {scores[:5]}")
