"""
Utility functions for visualization and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import config


def plot_confusion_matrix(cm, labels=['Normal', 'Attack'], title='Confusion Matrix', save_path=None):
    """Plot a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_score_distribution(normal_scores, attack_scores, title='Anomaly Score Distribution',
                            save_path=None):
    """Plot distribution of anomaly scores for normal vs attack traffic."""
    plt.figure(figsize=(10, 6))

    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    plt.hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='red', density=True)

    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Score distribution saved to {save_path}")
    plt.close()


def plot_feature_importance(importance_scores, feature_names, top_n=20, save_path=None):
    """Plot feature importance."""
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices]
    sorted_scores = importance_scores[indices]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_scores)), sorted_scores, align='center')
    plt.yticks(range(len(sorted_scores)), sorted_names)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))

    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    plt.close()


def generate_report(results, preprocessor, save_path=None):
    """Generate a text report of model performance."""
    report_lines = [
        "=" * 60,
        "NETWORK ANOMALY DETECTION REPORT",
        "=" * 60,
        "",
        "Model Comparison:",
        "-" * 40,
    ]

    for model_name in ['Autoencoder', 'Isolation Forest', 'Ensemble']:
        if model_name in results:
            r = results[model_name]
            # Get confusion matrix values
            cm = r['confusion_matrix']
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            report_lines.extend([
                f"\n{model_name}:",
                f"  Accuracy:  {r['accuracy']:.4f}",
                f"  Precision: {r['precision']:.4f}",
                f"  Recall:    {r['recall']:.4f}",
                f"  F1-Score:  {r['f1']:.4f}",
                f"  AUC-ROC:   {r['auc']:.4f}",
                f"  FPR:       {fpr:.4f}",
            ])

    report_lines.extend([
        "",
        "=" * 60,
        "Confusion Matrices:",
        "=" * 60,
    ])

    for model_name in ['Autoencoder', 'Isolation Forest', 'Ensemble']:
        if model_name in results:
            cm = results[model_name]['confusion_matrix']
            report_lines.extend([
                f"\n{model_name}:",
                f"  True Negatives:  {cm[0,0]}",
                f"  False Positives: {cm[0,1]}",
                f"  False Negatives: {cm[1,0]}",
                f"  True Positives:  {cm[1,1]}",
            ])

    report_lines.extend([
        "",
        "=" * 60,
        "",
    ])

    report_text = "\n".join(report_lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {save_path}")

    print(report_text)
    return report_text


def save_results_summary(results, save_path=None):
    """Save results to CSV for easy comparison."""
    rows = []
    for model_name in ['Autoencoder', 'Isolation Forest', 'Ensemble']:
        if model_name in results:
            r = results[model_name]
            # Get FPR from confusion matrix
            cm = r['confusion_matrix']
            tn, fp = cm[0, 0], cm[0, 1]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            rows.append({
                'Model': model_name,
                'Accuracy': r['accuracy'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1-Score': r['f1'],
                'AUC-ROC': r['auc'],
                'FPR': fpr
            })

    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results summary saved to {save_path}")

    return df


def analyze_errors(X_test, y_test, y_pred, feature_names, normal_class, save_path=None):
    """Analyze false positives and false negatives.

    Args:
        X_test: Test features
        y_test: Test labels (encoded)
        y_pred: Predictions (0=normal, 1=attack)
        feature_names: List of feature names
        normal_class: The encoded label for normal/benign traffic
        save_path: Optional path to save analysis
    """
    y_binary = (y_test != normal_class).astype(int)

    fp_mask = (y_binary == 0) & (y_pred == 1)  # False positives
    fn_mask = (y_binary == 1) & (y_pred == 0)  # False negatives

    analysis = {
        'false_positives': {
            'count': fp_mask.sum(),
            'percentage': fp_mask.mean() * 100
        },
        'false_negatives': {
            'count': fn_mask.sum(),
            'percentage': fn_mask.mean() * 100
        }
    }

    print("\nError Analysis:")
    print(f"  False Positives: {analysis['false_positives']['count']} ({analysis['false_positives']['percentage']:.2f}%)")
    print(f"  False Negatives: {analysis['false_negatives']['count']} ({analysis['false_negatives']['percentage']:.2f}%)")

    return analysis
