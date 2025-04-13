import math

import numpy as np
import pandas as pd

"""
Notes:
assengerid: numpy.int64
survived: numpy.int64
pclass: numpy.int64
name: string
sex: string
age: numpy.float64
sibsp: numpy.int64
parch: numpy.int64
ticket: string
fare: numpy.float64
cabin: float
embarked: string
"""


def fetch_and_format_kaggle_titanic_data():
    # Read the training dataset specified
    dataset = pd.read_csv('../data/train.csv')

    # Format feature/label names to all lowercase letters
    dataset.columns = [ft.lower() for ft in dataset.columns]

    # Read the training dataset specified
    labels = pd.DataFrame(data=dataset['survived'].values)

    # Remove target from feature dataset
    features = dataset.drop('survived', axis=1)

    # Convert categorical features into numerical features
    features['sex'] = features['sex'].map({'male': 0, 'female': 1}).astype(int)
    features['cabin'] = features['cabin'].notnull().astype(int)

    # Drop below features for now
    features = features.drop(labels=['name', 'ticket', 'embarked'], axis=1)
    return features, labels


def split_data(features, labels, test_percent=0.2, random_state=2025):
    if isinstance(features, pd.DataFrame):
        features = features.values

    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    num_examples = features.shape[0]

    np.random.default_rng(random_state)
    shuffled_indices = np.random.permutation(num_examples)
    shuffled_X = features[shuffled_indices]
    shuffled_y = labels[shuffled_indices]

    test_size = math.ceil(test_percent * num_examples)

    features_train = shuffled_X[test_size:, :-1]
    features_test = shuffled_X[:test_size, :-1]

    labels_train = shuffled_y[test_size:].reshape((-1,)).astype(int)
    labels_test = shuffled_y[:test_size].reshape((-1,)).astype(int)

    return features_train, labels_train, features_test, labels_test
