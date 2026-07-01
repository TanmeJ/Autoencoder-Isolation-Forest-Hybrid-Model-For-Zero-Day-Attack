"""
Network Anomaly Detection Mini Project

This project uses two simple anomaly detection ideas:
1. Autoencoder reconstruction error
2. Isolation Forest anomaly score

The final prediction is made by averaging both scores.

Usage:
    python main.py train
    python main.py evaluate
    python main.py predict data.csv
"""
import argparse
import os
import sys

import numpy as np

import config
from data_preprocessing import DataPreprocessor
from ensemble_model import EnsembleAnomalyDetector
from utils import generate_report, plot_confusion_matrix, save_results_summary


def print_results(results):
    """Print the main evaluation values in a simple table."""
    print("\nResults")
    print("-" * 72)
    print(f"{'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 72)
    for name in ["Autoencoder", "Isolation Forest", "Ensemble"]:
        if name in results:
            row = results[name]
            print(
                f"{name:<18} "
                f"{row['accuracy']:>10.4f} "
                f"{row['precision']:>10.4f} "
                f"{row['recall']:>10.4f} "
                f"{row['f1']:>10.4f}"
            )
    print("-" * 72)


def train_mode(args):
    """Load data, train both models, test them, and save the output."""
    config.RANDOM_STATE = args.seed
    config.set_random_seeds(args.seed)

    print("=" * 60)
    print("NETWORK ANOMALY DETECTION MINI PROJECT")
    print("Hybrid model: Autoencoder + Isolation Forest")
    print("=" * 60)

    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()

    model = EnsembleAnomalyDetector(
        autoencoder_weight=config.ENSEMBLE_PARAMS["autoencoder_weight"],
        isolation_forest_weight=config.ENSEMBLE_PARAMS["isolation_forest_weight"],
    )
    model.train(
        data["X_train"],
        data["y_train"],
        data["normal_class"],
        data["X_val"],
    )

    results = model.evaluate(
        data["X_test"],
        data["y_test"],
        data["normal_class"],
    )

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    model.save_models()
    preprocessor.save_state()

    confusion_matrix_outputs = [
        ("Autoencoder", "Autoencoder Confusion Matrix", "confusion_matrix_autoencoder.png"),
        ("Isolation Forest", "Isolation Forest Confusion Matrix", "confusion_matrix_isolation_forest.png"),
        ("Ensemble", "Hybrid Model Confusion Matrix", "confusion_matrix_ensemble.png"),
    ]
    for model_name, title, filename in confusion_matrix_outputs:
        plot_confusion_matrix(
            results[model_name]["confusion_matrix"],
            title=title,
            save_path=os.path.join(config.RESULTS_DIR, filename),
        )

    generate_report(results, preprocessor, save_path=os.path.join(config.RESULTS_DIR, "report.txt"))
    save_results_summary(results, save_path=os.path.join(config.RESULTS_DIR, "results_summary.csv"))

    print_results(results)
    print(f"\nSaved models to: {config.MODEL_DIR}")
    print(f"Saved results to: {config.RESULTS_DIR}")


def evaluate_mode(args):
    """Evaluate saved models on the dataset again."""
    needed_files = [
        "autoencoder.h5",
        "isolation_forest.pkl",
        "isolation_forest_scaler.pkl",
        "preprocessor.npz",
    ]
    missing = [name for name in needed_files if not os.path.exists(os.path.join(args.model_dir, name))]
    if missing:
        print("Missing saved files:", ", ".join(missing))
        print("Run 'python main.py train' first.")
        sys.exit(1)

    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()

    model = EnsembleAnomalyDetector()
    model.load_models(args.model_dir)

    results = model.evaluate(
        data["X_test"],
        data["y_test"],
        data["normal_class"],
    )
    print_results(results)


def predict_mode(args):
    """Use saved models to make predictions on a new CSV file."""
    preprocessor = DataPreprocessor()
    preprocessor.load_state(args.model_dir)

    model = EnsembleAnomalyDetector()
    model.load_models(args.model_dir)

    preprocessor.data_path = args.file
    df = preprocessor.clean_data(preprocessor.load_data())

    X = df[preprocessor.feature_columns].values
    X = preprocessor.scaler.transform(X)

    scores = model.predict_anomaly_scores(X)
    predictions = model.predict(X)

    output = df.copy()
    output["hybrid_score"] = scores
    output["prediction"] = np.where(predictions == 1, "Suspicious", "Normal")

    output_path = args.file.replace(".csv", "_predictions.csv")
    output.to_csv(output_path, index=False)

    suspicious_count = int(predictions.sum())
    normal_count = len(predictions) - suspicious_count

    print("\nPrediction Summary")
    print("-" * 40)
    print(f"Rows checked: {len(predictions)}")
    print(f"Normal rows: {normal_count}")
    print(f"Suspicious rows: {suspicious_count}")
    print(f"Saved file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Network anomaly detection mini project")
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train", help="Train and test the hybrid model")
    train_parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate saved models")
    evaluate_parser.add_argument("--model-dir", default=config.MODEL_DIR)

    predict_parser = subparsers.add_parser("predict", help="Predict on a CSV file")
    predict_parser.add_argument("file")
    predict_parser.add_argument("--model-dir", default=config.MODEL_DIR)

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "evaluate":
        evaluate_mode(args)
    elif args.mode == "predict":
        predict_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
