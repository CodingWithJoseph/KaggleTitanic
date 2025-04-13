import pandas as pd
import numpy as np


class XGBoostModel:
    pass


class BoostedTree:
    def __init__(self, X, gradients, hessians, max_depth, example_indices, params):
        # Set feature space
        self.X = X.values if isinstance(X, pd.DataFrame) else X

        # Set gradients
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients

        # Set hessians
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians

        # Set maximum number of levels below current node
        self.max_depth = max_depth

        # Example indices used for row sampling
        self.example_indices = example_indices

        # Set number of examples
        self.num_examples = len(example_indices)

        # Set number of features
        self.num_features = X.shape[1]

        # Set split feature index
        self.split_feature = 0

        # Set split feature threshold value
        self.threshold = 0

        # Set lambda used in regularization
        self.regularization_lambda = params['lambda'] if params['lambda'] else 1.0

        # Set gamma used in regularization
        self.gamma = params['gamma'] if params['gamma'] else 1.0

        # Set split score to -inf: higher is better
        self.split_score = 0.0

    def _find_best_split(self, fidx):
        feature = self.X[self.example_indices, fidx]
        sorted_idx = np.argsort(feature)

        sorted_feature = feature[sorted_idx]
        sorted_gradients = self.gradients[sorted_idx]
        sorted_hessians = self.hessians[sorted_idx]

        gradient_sum = sorted_gradients.sum()
        hessian_sum = sorted_hessians.sum()

        right_gradient_sum = gradient_sum
        right_hessian_sum = hessian_sum

        left_gradient_sum = 0.0
        left_hessian_sum = 0.0

        for ex in range(self.num_examples):
            left_gradient_sum += sorted_gradients[ex]
            left_hessian_sum += sorted_hessians[ex]

            right_gradient_sum -= sorted_gradients[ex]
            right_hessian_sum -= sorted_gradients[ex]

            left_score = left_gradient_sum / (left_hessian_sum + self.regularization_lambda)
            right_score = right_gradient_sum / (right_hessian_sum + self.regularization_lambda)
            score_before_split = gradient_sum / (hessian_sum + self.regularization_lambda)

            gain = 0.5 * (left_score + right_score - score_before_split) - self.gamma

            if gain > self.split_score:
                self.threshold = (sorted_feature[ex] + sorted_feature[ex + 1]) / 2
                self.split_index = sorted_idx[ex]
                self.split_score = gain

    def _build_structure(self):
        if self.max_depth <= 0.0:
            return

        for ft in range(self.num_features):
            self._find_best_split(ft)

        if self.is_leaf:
            return

        left_data = np.where(self.X)
        right_data = np.where(self.X )

    def _predict_row(self, X):
        pass

    def predict(self, X):
        pass

    @property
    def is_leaf(self): return self.split_score == 0


if __name__ == '__main__':
    array = np.array([[1, 2, 3, 4, 5, 6, 7],
                      [8, 9, 10, 11, 12, 13, 14],
                      [15, 16, 17, 18, 19, 20, 21],
                      [22, 23, 24, 25, 26, 27, 28],
                      [29, 30, 31, 32, 33, 34, 35]])

    for col in range(array.shape[1]):
        print(array[:, col])
