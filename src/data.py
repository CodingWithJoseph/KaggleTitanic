import math
import numpy as np
import pandas as pd


def fetch_titanic_data(path='../data/train.csv', target_name=None, as_X_y=False):
    # Load dataset from CSV
    dataset = pd.read_csv(path)

    if as_X_y:
        assert target_name is not None, "Must set target_name when X_y is true"
        labels = pd.DataFrame(data=dataset[target_name].values)  # Extract target
        dataset = dataset.drop(target_name, axis=1)  # Drop target from features
        return dataset, labels

    return dataset


def clean_column_names(features):
    # Lowercase all column names
    return [ft.lower() for ft in features.columns]


def preprocess_features(features, categorical_mapping=None, binary_from_null=None, drop_columns=None):
    # Map categorical columns to integers
    if categorical_mapping:
        for col, mapping in categorical_mapping.items():
            if col in features.columns:
                features[col] = features[col].map(mapping).astype(int)

    # Convert null presence to binary flags
    if binary_from_null:
        for col in binary_from_null:
            if col in features.columns:
                features[col] = features[col].notnull().astype(int)

    # Drop irrelevant columns
    if drop_columns:
        features = features.drop(columns=drop_columns, errors='ignore')

    return features


def split_data(features, labels, test_percent=0.2, random_state=2025):
    # Convert pandas DataFrames to NumPy arrays if necessary
    if isinstance(features, pd.DataFrame):
        features = features.values

    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    # Get number of data points (rows)
    num_examples = features.shape[0]

    # Shuffle the dataset using the given random seed
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(num_examples)
    shuffled_X = features[shuffled_indices]
    shuffled_y = labels[shuffled_indices]

    # Calculate number of test examples based on given test size percent
    test_size = math.ceil(test_percent * num_examples)

    # Split features: everything after the test_size is training
    X_train = shuffled_X[test_size:]
    X_test = shuffled_X[:test_size]

    # Split labels: reshape to ensure they're 1D and cast to int
    y_train = shuffled_y[test_size:].ravel().astype(int)
    y_test = shuffled_y[:test_size].ravel().astype(int)

    # Return split datasets
    return X_train, X_test, y_train, y_test
