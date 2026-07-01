"""
Basic validation tests for Zero-Day Attack Detection System

This file contains simple tests to verify the pipeline works correctly.
Run with: python test_pipeline.py
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def test_config():
    """Test that config values are reasonable."""
    print("Testing config...")

    # Check random seed is set
    assert config.RANDOM_STATE == 42, "Random seed should be 42"

    # Check test sizes are valid
    assert 0 < config.TEST_SIZE < 1, "Test size should be between 0 and 1"
    assert 0 < config.VALIDATION_SIZE < 1, "Validation size should be between 0 and 1"

    # Check ensemble weights sum to 1 (roughly)
    total_weight = config.ENSEMBLE_PARAMS['autoencoder_weight'] + config.ENSEMBLE_PARAMS['isolation_forest_weight']
    assert abs(total_weight - 1.0) < 0.01, "Ensemble weights should sum to 1"

    print("  Config tests passed!")
    return True


def test_data_preprocessing():
    """Test data preprocessing functions."""
    print("\nTesting data preprocessing...")
    from data_preprocessing import DataPreprocessor

    # Create preprocessor
    preprocessor = DataPreprocessor()

    # Test that scaler is initialized
    assert preprocessor.scaler is not None, "Scaler should be initialized"
    assert preprocessor.label_encoder is not None, "Label encoder should be initialized"

    print("  Data preprocessing tests passed!")
    return True


def test_autoencoder_model():
    """Test autoencoder model creation."""
    print("\nTesting autoencoder model...")
    from autoencoder_model import AutoencoderAnomalyDetector

    # Create model with dummy input dimension
    input_dim = 10
    detector = AutoencoderAnomalyDetector(input_dim=input_dim)

    # Build model
    model = detector.build_model()

    # Check model structure
    assert model is not None, "Model should be created"
    assert model.input_shape[1] == input_dim, "Input dimension should match"
    assert model.output_shape[1] == input_dim, "Output dimension should match input"

    print("  Autoencoder model tests passed!")
    return True


def test_isolation_forest_model():
    """Test isolation forest model creation."""
    print("\nTesting isolation forest model...")
    from isolation_forest_model import IsolationForestDetector

    # Create detector
    detector = IsolationForestDetector()

    # Build model
    model = detector.build_model()

    # Check model is created
    assert model is not None, "Model should be created"

    print("  Isolation Forest model tests passed!")
    return True


def test_ensemble_model():
    """Test ensemble model creation."""
    print("\nTesting ensemble model...")
    from ensemble_model import EnsembleAnomalyDetector

    # Create ensemble with default weights
    ensemble = EnsembleAnomalyDetector()

    # Check weights are normalized
    assert abs(ensemble.autoencoder_weight + ensemble.isolation_forest_weight - 1.0) < 0.01, "Weights should sum to 1"

    # Test with custom weights
    ensemble2 = EnsembleAnomalyDetector(
        autoencoder_weight=0.7,
        isolation_forest_weight=0.3
    )
    assert abs(ensemble2.autoencoder_weight - 0.7) < 0.01, "Autoencoder weight should be 0.7"

    print("  Ensemble model tests passed!")
    return True


def test_with_sample_data():
    """Test pipeline with sample data."""
    print("\nTesting pipeline with sample data...")
    from autoencoder_model import AutoencoderAnomalyDetector
    from isolation_forest_model import IsolationForestDetector

    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)  # 100 samples, 10 features
    X_test = np.random.randn(20, 10)     # 20 test samples

    # Test Autoencoder - use verbose=0 to suppress training output
    ae = AutoencoderAnomalyDetector(input_dim=10)
    ae.build_model()
    ae.compile_model()
    ae.params['epochs'] = 2  # Reduce epochs for testing

    # Quick training (just a few epochs for test)
    ae.train(X_train, X_train)  # Train on itself for test
    ae.set_threshold(X_train)

    scores = ae.predict_anomaly_scores(X_test)
    assert len(scores) == 20, "Should get 20 scores"
    assert scores.min() >= 0, "Scores should be non-negative"

    predictions = ae.predict(X_test)
    assert len(predictions) == 20, "Should get 20 predictions"
    assert set(predictions).issubset({0, 1}), "Predictions should be 0 or 1"

    # Test Isolation Forest
    if_detector = IsolationForestDetector()
    if_detector.train(X_train)
    if_detector.set_threshold(X_train)

    scores = if_detector.predict_anomaly_scores(X_test)
    assert len(scores) == 20, "Should get 20 scores"
    # Scores can be outside [0,1] if test data differs from training
    # Just verify they're reasonable numbers
    assert not np.isnan(scores).any(), "Scores should not contain NaN"
    assert not np.isinf(scores).any(), "Scores should not contain infinity"

    predictions = if_detector.predict(X_test)
    assert len(predictions) == 20, "Should get 20 predictions"

    print("  Sample data tests passed!")
    return True


def test_reproducibility():
    """Test that random seeds produce reproducible results."""
    print("\nTesting reproducibility...")
    from isolation_forest_model import IsolationForestDetector

    # Set seed and create first model
    config.set_random_seeds(42)
    np.random.seed(42)
    X = np.random.randn(50, 5)

    detector1 = IsolationForestDetector()
    detector1.train(X)
    scores1 = detector1.predict_anomaly_scores(X[:5])

    # Reset and create second model
    config.set_random_seeds(42)
    np.random.seed(42)
    X = np.random.randn(50, 5)

    detector2 = IsolationForestDetector()
    detector2.train(X)
    scores2 = detector2.predict_anomaly_scores(X[:5])

    # Check results are identical
    assert np.allclose(scores1, scores2), "Same seed should produce same results"

    print("  Reproducibility tests passed!")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("Running Validation Tests")
    print("=" * 60)

    tests = [
        test_config,
        test_data_preprocessing,
        test_autoencoder_model,
        test_isolation_forest_model,
        test_ensemble_model,
        test_with_sample_data,
        test_reproducibility
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)