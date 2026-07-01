<<<<<<< HEAD
# Network Anomaly Detection Mini Project

This is a second-year software engineering mini project that uses machine
learning to mark network traffic rows as normal or suspicious.

The project uses a hybrid approach:

- **Autoencoder**: learns to recreate normal traffic data. A high reconstruction
  error can mean the row is unusual.
- **Isolation Forest**: finds rows that look different from the rest of the data.
- **Hybrid score**: averages both model scores and uses a threshold to decide
  whether a row is suspicious.

This is a learning project, not a production security tool.

## How To Run

Install the libraries:

```bash
pip install -r requirements.txt
```

Train and test the model:

```bash
python main.py train
```

Evaluate saved models:

```bash
python main.py evaluate
```

Predict on another CSV file:

```bash
python main.py predict your_file.csv
```

## Project Files

```text
cn/
|-- main.py                    # Main program
|-- config.py                  # Settings used by the project
|-- data_preprocessing.py      # Loads and prepares the dataset
|-- autoencoder_model.py       # Autoencoder model
|-- isolation_forest_model.py  # Isolation Forest model
|-- ensemble_model.py          # Combines both models
|-- utils.py                   # Graphs and report helpers
|-- test_pipeline.py           # Basic tests
|-- requirements.txt           # Python libraries
|-- models/                    # Saved model files
|-- results/                   # Output graphs and results
`-- Friday-WorkingHours-Morning.pcap_ISCX.csv
```

## Basic Steps

1. Load the CSV dataset.
2. Clean missing and infinite values.
3. Split the data into training, validation, and test sets.
4. Scale the numeric features.
5. Train the autoencoder on normal traffic.
6. Train the Isolation Forest.
7. Average both anomaly scores.
8. Print accuracy, precision, recall, F1-score, and a confusion matrix.

## Dataset

The project expects a CICIDS-style CSV file with:

- numeric network traffic columns;
- a `Label` column;
- a normal class such as `BENIGN`.

## Configuration

The main settings are in `config.py`.

```python
AUTOENCODER_PARAMS = {
    "encoding_dims": [32, 16],
    "latent_dim": 8,
    "epochs": 10
}

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 50,
    "contamination": 0.1
}

ENSEMBLE_PARAMS = {
    "autoencoder_weight": 0.5,
    "isolation_forest_weight": 0.5,
    "anomaly_percentile": 95
}
```

## Metrics Used

- **Accuracy**: total correct predictions.
- **Precision**: how many predicted suspicious rows were actually attacks.
- **Recall**: how many attacks were found.
- **F1-score**: balance between precision and recall.
- **False positive rate**: normal rows incorrectly marked suspicious.

For this type of project, accuracy alone can be misleading because most rows in
the dataset may be normal.

## Limitations

- The model is trained and tested on one dataset.
- It only gives normal/suspicious output, not the exact attack type.
- Thresholds are simple percentile-based values.
- Results can change if a different dataset is used.
- More testing is needed before using this for real security work.

## Future Work

- Try different threshold values.
- Compare the hybrid model against each individual model.
- Add attack type classification.
- Improve charts and error analysis.
- Test with more network traffic datasets.
=======
# Autoencoder-Isolation-Forest-Hybrid-Model-For-Zero-Day-Attack
A hybrid machine learning model for zero-day attack detection using Autoencoder and Isolation Forest. The project analyzes network flow data, combines reconstruction and anomaly scores, and classifies traffic as Normal or Suspicious. Built with Python, TensorFlow/Keras, and Scikit-learn.
>>>>>>> 7ba18f2477e0b697c1cf7f0e36f32d9fd3181281
