import math
import numpy as np
import pandas as pd


def fetch_titanic_data(path='../data/train.csv', target_name=None, as_X_y=False):
    # Load dataset from CSV
    X = pd.read_csv(path)

    if as_X_y:
        assert target_name, "Must set target_name when X_y is true"
        y = X.pop(target_name)  # Extract target
        return X, y

    return X


def clean_column_names(X):
    # Lowercase all column names
    X.columns = X.columns.str.lower()


def preprocess_features(X, categorical_mapping=None, binary_from_null=None, drop_columns=None):
    # Map categorical columns to integers
    if categorical_mapping:
        for col, mapping in categorical_mapping.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).astype(int)

    # Convert null presence to binary flags
    if binary_from_null:
        for col in binary_from_null:
            if col in X.columns:
                X[col] = X[col].notnull().astype(int)

    # Drop irrelevant columns
    X.drop(columns=[col for col in drop_columns if drop_columns and col in X.columns], inplace=True)

    return X


def prepare_data(X, categorical_mapping=None, binary_from_null=None, drop_columns=None, fill_with_median=None):
    clean_column_names(X)
    X = preprocess_features(X, categorical_mapping, binary_from_null, drop_columns)
    if fill_with_median:
        X[fill_with_median] = X[fill_with_median].fillna(X[fill_with_median].median())
    return X


def split_data(X, y, test_percent=0.2, random_state=2025):
    # Convert pandas DataFrames to NumPy arrays if necessary
    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(y, pd.DataFrame):
        y = y.values

    # Get number of data points (rows)
    num_examples = X.shape[0]

    # Shuffle the dataset using the given random seed
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(num_examples)
    shuffled_X = X[shuffled_indices]
    shuffled_y = y[shuffled_indices]

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
