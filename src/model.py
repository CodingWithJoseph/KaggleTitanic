import math
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


class XGBoost:
    def __init__(self, params, objective, seed=42):
        # Boosted Trees
        self.trees = []

        # Wrap params with defaultdict prevent key access errors
        self.params = defaultdict(lambda: None, params)

        # Objective function, loss, gradient, hessian
        self.objective = objective

        # Subsample percent for rows: use the entire set when 1.0
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0

        # Base score for predictions
        self.base_score = self.params['base_score'] if self.params['base_score'] else 0.5

        # Learning rate
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 1e-1

        # Max depth below this node
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5

        # Random number generator used for sampling randomly
        self.rng = np.random.default_rng(seed=seed)

    def fit(self, X, y, num_rounds):
        # Make base predictions for model
        predictions = self.base_score * np.ones(y.shape)

        for rnd in range(num_rounds):
            # Compute gradients from previous predictions
            gradients = self.objective.gradients(y, predictions)

            # Compute hessians, 2nd order, from previous predictions
            hessians = self.objective.hessians(y, predictions)

            # Compute row subsample if applicable else None
            idxs = None if self.subsample == 1.0 else self.rng.choice(
                a=len(y),
                size=math.floor(self.subsample * len(y)),
                replace=False
            )

            # Create new tree and add it to ensemble
            tree = BoostedTree(
                X=X,
                gradients=gradients,
                hessians=hessians,
                params=self.params,
                max_depth=self.max_depth,
                idxs=idxs
            )
            self.trees.append(tree)

            # Make and update predictions
            predictions += self.learning_rate * tree.predict(X)

            # Report current loss
            print(f'Boost Round: [{rnd}] ----> Train Loss = {self.objective.loss(y, predictions)}')

    def predict(self, X):
        return self.base_score + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)


class BoostedTree:
    def __init__(self, X, gradients, hessians, params, max_depth, idxs=None):
        # Convert X to numpy array and set
        self.X = X.values if isinstance(X, pd.DataFrame) else X

        # Convert gradients to numpy array and set
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients

        # Convert hessians to numpy array and set
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians

        # Params should already be wrapped in a defaultdict
        self.params = params

        # Regularization scalar applied to weights
        self.min_child_weight = self.params['min_child_weight'] if self.params['min_child_weight'] else 1.0

        # Regularization scalar applied to weights
        self._lambda = self.params['reg_lambda'] if self.params['reg_lambda'] else 1.0

        # Regularization scalar applied to number of leafs
        self.gamma = self.params['gamma'] if self.params['gamma'] else 0.0

        # Regularization scalar applied to number of leafs
        self.column_subsample = self.params['column_subsample'] if self.params['column_subsample'] else 1.0

        self.max_depth = max_depth

        self.ridxs = idxs if idxs is not None else np.arange(len(gradients))

        # Compute and set number of examples
        self.num_examples = len(self.ridxs)

        # Compute and set number of features
        self.num_features = X.shape[1]

        # The weight for current node but only used if is_leaf is true
        self.weight = -self.gradients[self.ridxs].sum() / (self.hessians[self.ridxs].sum() + self._lambda)

        # Best split score, best computed score for sampled feature and threshold combinations
        self.split_score = 0.0

        # The feature that was used to split the node
        self.split_idx = 0

        # The value for a given feature used to split the node
        self.threshold = 0.0

        self._build_tree_structure()

    def _build_tree_structure(self):
        if self.max_depth <= 0:
            return

        for fidx in range(self.num_features):
            self._find_best_split_score(fidx)

        # If is_leaf is true after trying sampled splits then return
        if self._is_leaf:
            return

        # Retrieve the chosen feature data
        feature = self.X[self.ridxs, self.split_idx]

        # Split chosen feature data
        left_idxs = np.nonzero(feature <= self.threshold)[0]
        right_idxs = np.nonzero(feature > self.threshold)[0]

        # Create left and right splits
        self.left = BoostedTree(self.X, self.gradients, self.hessians, self.params, self.max_depth - 1, left_idxs)
        self.right = BoostedTree(self.X, self.gradients, self.hessians, self.params, self.max_depth - 1, right_idxs)

    def _find_best_split_score(self, fidx):
        feature = self.X[self.ridxs, fidx]
        sorted_idxs = np.argsort(feature)

        sorted_feature = feature[sorted_idxs]
        sorted_gradient = self.gradients[sorted_idxs]
        sorted_hessians = self.hessians[sorted_idxs]

        hessian_sum = sorted_hessians.sum()
        gradient_sum = sorted_gradient.sum()

        right_hessian_sum = hessian_sum
        right_gradient_sum = gradient_sum

        left_hessian_sum = 0.0
        left_gradient_sum = 0.0

        for idx in range(0, self.num_examples - 1):
            candidate = sorted_feature[idx]
            neighbor = sorted_feature[idx + 1]

            gradient = sorted_gradient[idx]
            hessian = sorted_hessians[idx]

            right_gradient_sum -= gradient
            right_hessian_sum -= hessian

            left_gradient_sum += gradient
            left_hessian_sum += hessian

            if right_hessian_sum <= self.min_child_weight:
                return

            right_score = (right_gradient_sum ** 2) / (right_hessian_sum + self._lambda)
            left_score = (left_gradient_sum ** 2) / (left_hessian_sum + self._lambda)
            score_before_split = (gradient_sum ** 2) / (hessian_sum + self._lambda)

            gain = 0.5 * (left_score + right_score - score_before_split) - self.gamma / 2

            if gain > self.split_score:
                self.split_score = gain
                self.split_idx = fidx
                self.threshold = (candidate + neighbor) / 2

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict_row(example) for example in X[:]])

    def _predict_row(self, example):
        if self._is_leaf:
            return self.weight

        child = self.left if example[self.split_idx] <= self.threshold else self.right
        return child._predict_row(example)

    @property
    def _is_leaf(self):
        return self.split_score == 0.0


class SquaredErrorObjective:
    @staticmethod
    def loss(labels, predictions): return np.mean((labels - predictions) ** 2)

    @staticmethod
    def gradients(labels, predictions): return predictions - labels

    @staticmethod
    def hessians(labels, predictions): return np.ones(len(labels))


if __name__ == '__main__':
    data, true = fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, true, test_size=0.3, random_state=43)

    hyperparameters = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'reg_lambda': 1.5,
        'lambda': 1.5,
        'gamma': 0.0,
        'min_child_weight': 25,
        'base_score': 0.0,
        'tree_method': 'exact',
    }
    num_boost_round = 50

    # train the from-scratch XGBoost model
    model = XGBoost(hyperparameters, SquaredErrorObjective(), seed=42)
    model.fit(X_train, y_train, num_boost_round)

    # train the library XGBoost model
    baseline = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model_xgb = xgb.train(hyperparameters, baseline, num_boost_round)

    output = model.predict(X_test)
    output_xgb = model_xgb.predict(dtest)
    print(f'scratch score: {SquaredErrorObjective().loss(y_test, output)}')
    print(f'xgboost score: {SquaredErrorObjective().loss(y_test, output_xgb)}')
