import math
from collections import defaultdict

import numpy as np


class XGBoost:
    def __init__(self, objective, params, seed=42):
        # Objective function: Loss, Gradient, and Hessian functions
        self.boosted_trees = []

        # Objective function: Loss, Gradient, and Hessian functions
        self.objective = objective

        # Random number generator used for random indices
        self.rng = np.random.default_rng(seed=seed)

        # Model hyperparameters
        self.params = defaultdict(lambda: None, params)

        # Initial predictions for additive model
        self.base_prediction = self.params['base_prediction'] if self.params['base_prediction'] else 0.5

        # Row subsampling percent
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0

    def fit(self, X, y, num_boost_rounds, verbose=False):
        predictions = self.base_prediction * np.ones(shape=y.shape)

        for rnd in range(num_boost_rounds):
            # Calculate gradients for current tree
            gradients = self.objective.gradient(y, predictions)

            # Calculate hessians for current tree
            hessians = self.objective.hessians(y, predictions)

            # Randomly select subsample examples
            ex_idx = None if self.subsample is None else self.rng.choice(
                a=len(y),
                size=math.floor(self.subsample * len(y)),
                replace=False
            )

        if verbose:
            print(f"Boost Round: {rnd}; Loss: {self.objective.loss(y, predictions)}")


def predict(self, X):
    pass
