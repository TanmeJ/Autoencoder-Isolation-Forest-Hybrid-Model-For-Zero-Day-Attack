"""
Configuration file for Zero-Day Attack Detection System
"""

import os
import random
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Friday-WorkingHours-Morning.pcap_ISCX.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42


def set_random_seeds(seed=RANDOM_STATE):
    """Set all random seeds for reproducibility.

    Call this at the start of your script to ensure reproducible results.
    This sets seeds for: Python random, NumPy, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Enable deterministic operations (may slow down but ensures reproducibility)
        tf.config.experimental.enable_op_determinism()
    except (ImportError, AttributeError):
        pass  # TensorFlow not installed or old version


# Data Processing
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature Engineering
# Columns to drop (non-numeric or identifier columns)
DROP_COLUMNS = [
    'Flow ID',
    'Source IP',
    'Source Port',
    'Destination IP',
    'Destination Port',
    'Protocol',
    'Timestamp',
    'External IP',
    'Initiated',
    'flow_start',
    'flow_end'
]

# Target column name (label)
LABEL_COLUMN = 'Label'

# Autoencoder Parameters
AUTOENCODER_PARAMS = {
    'encoding_dims': [32, 16],  # Small network for a student project
    'latent_dim': 8,
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 10,
    'early_stopping_patience': 3
}

# Isolation Forest Parameters
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 50,
    'contamination': 0.1,  # Expected proportion of anomalies
    'max_samples': 'auto',
    'max_features': 1.0,
    'random_state': RANDOM_STATE
}

# Ensemble Parameters
ENSEMBLE_PARAMS = {
    'autoencoder_weight': 0.5,
    'isolation_forest_weight': 0.5,
    'threshold_method': 'percentile',
    'anomaly_percentile': 95
}

# Evaluation
EVALUATION_PARAMS = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'confusion_matrix': True,
    'roc_curve': True
}

# Version tracking
MODEL_VERSION = "1.0.0"
CONFIG_HASH = None  # Set at runtime for reproducibility tracking


def get_config_hash():
    """Generate a hash of current configuration for tracking."""
    import hashlib
    config_str = f"{RANDOM_STATE}_{TEST_SIZE}_{AUTOENCODER_PARAMS}_{ISOLATION_FOREST_PARAMS}_{ENSEMBLE_PARAMS}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
