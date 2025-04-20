import os.path
import pandas as pd
from model_utils import open_model, save_model
from src.model import XGBoostSigmoid
from src.data import fetch_titanic_data, clean_column_names, preprocess_features, split_data, prepare_data

# Model Name
model_name = "xgboost_titanic.pkl"

# Preprocessing configuration for the Titanic dataset
config = {
    'categorical_mapping': {'sex': {'male': 0, 'female': 1}},
    'binary_from_null': ['cabin'],
    'drop_columns': ['name', 'ticket', 'embarked']
}

# Hyperparameters for the XGBoost model
hyperparameters = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'reg_lambda': 1.5,
    'gamma': 3.0,
    'min_child_weight': 25,
    'base_score': 0.5,
    'tree_method': 'exact',
}

# Number of boosting rounds
num_boost_round = 50


def train(should_save):
    # Load Titanic data and split into features and labels
    X, y = fetch_titanic_data(target_name='Survived', as_X_y=True)

    # Apply feature transformations based on config
    X = prepare_data(X, **config)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train custom XGBoost model from scratch
    model = XGBoostSigmoid(hyperparameters, seed=42)
    model.train(X_train, y_train, num_boost_round)

    # Evaluate and print accuracy on test set
    print(f'My accuracy: {model.score(X_test, y_test)}')

    if should_save:
        print(f'Saving')
        save_model(model, model_name)


def kaggle(model):
    # Load Titanic data and split into features and labels
    X = fetch_titanic_data(path='../data/test.csv')

    passenger_ids = X['PassengerId'].values

    # Apply feature transformations based on config
    X = prepare_data(X, **config)

    predictions = model.predict(X, as_labels=True)

    output_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    output_df.to_csv(f'../submissions/submission_01.csv', index=False)


if __name__ == '__main__':
    if not os.path.exists(f'../models/{model_name}'):
        train(should_save=True)

    mdl, features = open_model(model_name)
    kaggle(mdl)
