"""
Data preprocessing module for CICIDS dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config


class DataPreprocessor:
    """Handles loading and preprocessing of CICIDS network traffic data."""

    def __init__(self, data_path=None):
        self.data_path = data_path or config.DATA_PATH
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.normal_class = None  # Store normal class index

    def load_data(self):
        """Load the CICIDS dataset."""
        print(f"Loading data from {self.data_path}...")
        try:
            df = pd.read_csv(self.data_path)
            print(f"Loaded {len(df)} samples with {len(df.columns)} features")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

    def clean_data(self, df):
        """Clean the dataset by handling missing values and infinite values."""
        print("Cleaning data...")

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Fill NaN values with column median (more robust than mean)
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Drop rows with remaining NaN (non-numeric columns)
        df = df.dropna()

        print(f"After cleaning: {len(df)} samples")
        return df

    def extract_features_labels(self, df):
        """Separate features and labels from the dataset."""
        print("Extracting features and labels...")

        # Find the label column (case-insensitive)
        label_col = None
        for col in df.columns:
            if col.lower().strip() == 'label':
                label_col = col
                break

        if label_col is None:
            # Try common alternatives
            for col in df.columns:
                if 'label' in col.lower():
                    label_col = col
                    break

        if label_col is None:
            raise ValueError("Could not find label column in dataset")

        print(f"Found label column: '{label_col}'")

        # Get features (drop non-numeric and identifier columns)
        columns_to_drop = []
        for col in df.columns:
            # Drop columns specified in config
            if col.strip() in config.DROP_COLUMNS:
                columns_to_drop.append(col)
            # Drop non-numeric columns except label
            elif col != label_col and df[col].dtype == 'object':
                columns_to_drop.append(col)

        # Also drop any columns that exist
        for drop_col in config.DROP_COLUMNS:
            for col in df.columns:
                if col.strip() == drop_col:
                    columns_to_drop.append(col)

        columns_to_drop = list(set(columns_to_drop))
        columns_to_drop = [col for col in columns_to_drop if col in df.columns and col != label_col]

        # Extract features
        feature_df = df.drop(columns=columns_to_drop)
        labels = feature_df[label_col].values
        feature_df = feature_df.drop(columns=[label_col])

        # Keep only numeric columns for features
        feature_df = feature_df.select_dtypes(include=[np.number])
        self.feature_columns = feature_df.columns.tolist()

        print(f"Features: {len(self.feature_columns)} columns")
        print(f"Unique labels: {np.unique(labels)}")

        return feature_df.values, labels, label_col

    def preprocess(self, test_size=None, validation_size=None):
        """Full preprocessing pipeline."""
        test_size = test_size or config.TEST_SIZE
        validation_size = validation_size or config.VALIDATION_SIZE

        # Load and clean data
        df = self.load_data()
        df = self.clean_data(df)

        # Extract features and labels
        X, y, label_col = self.extract_features_labels(df)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y_encoded
        )

        # Split train+val into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=validation_size/(1-test_size),
            random_state=config.RANDOM_STATE, stratify=y_train_val
        )

        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print(f"\nDataset splits:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        print(f"  Features: {X_train.shape[1]}")

        # Get normal vs attack distribution
        normal_class = None
        for i, cls in enumerate(self.label_encoder.classes_):
            if 'benign' in cls.lower() or 'normal' in cls.lower():
                normal_class = i
                break

        if normal_class is None:
            normal_class = 0  # Assume first class is normal

        # Store as attribute for later use
        self.normal_class = normal_class

        print(f"\nNormal class: {self.label_encoder.classes_[normal_class]}")
        print(f"Attack classes: {[c for i, c in enumerate(self.label_encoder.classes_) if i != normal_class]}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'normal_class': normal_class,
            'feature_columns': self.feature_columns,
            'n_features': X_train.shape[1]
        }

    def get_label_name(self, encoded_label):
        """Convert encoded label back to original name."""
        return self.label_encoder.inverse_transform([encoded_label])[0]

    def save_preprocessed_data(self, data_dict, path=None):
        """Save preprocessed data for later use."""
        path = path or config.MODEL_DIR
        np.save(os.path.join(path, 'X_train.npy'), data_dict['X_train'])
        np.save(os.path.join(path, 'X_test.npy'), data_dict['X_test'])
        np.save(os.path.join(path, 'y_train.npy'), data_dict['y_train'])
        np.save(os.path.join(path, 'y_test.npy'), data_dict['y_test'])
        print(f"Saved preprocessed data to {path}")

    def save_state(self, path=None):
        """Save preprocessor state for inference use."""
        path = path or config.MODEL_DIR
        np.savez(
            os.path.join(path, 'preprocessor.npz'),
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            label_encoder_classes=self.label_encoder.classes_,
            feature_columns=np.array(self.feature_columns),
            normal_class=np.array([self.normal_class])
        )
        print(f"Preprocessor state saved to {path}")

    def load_state(self, path=None):
        """Load preprocessor state for inference use."""
        path = path or config.MODEL_DIR
        state_path = os.path.join(path, 'preprocessor.npz')
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Preprocessor state not found at {state_path}")

        data = np.load(state_path, allow_pickle=True)
        self.scaler.mean_ = data['scaler_mean']
        self.scaler.scale_ = data['scaler_scale']
        self.label_encoder.classes_ = data['label_encoder_classes']
        self.feature_columns = list(data['feature_columns'])
        self.normal_class = int(data['normal_class'][0])

        print(f"Preprocessor state loaded from {path}")
        return self


if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()
    print("\nPreprocessing complete!")