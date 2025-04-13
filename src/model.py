import pandas as pd


class XGBoostModel:
    pass


class BoostedTree:
    def __init__(self, X, gradients, hessians, max_depth, params):
        # Set feature space
        self.X = X

        # Set gradients
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients

        # Set hessians
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians

        # Set maximum number of levels below current node
        self.max_depth = max_depth

        # Set split feature index
        self.split_feature = 0

        # Set split feature threshold value
        self.threshold = 0

        # Set lambda used in regularization
        self.regularization_lambda = params['lambda'] if params['lambda'] else 1.0

        # Set gamma used in regularization
        self.gamma = params['gamma'] if params['gamma'] else 1.0

    def _find_best_split(self, feature):
        pass

    def _build_structure(self):
        pass

    def _predict_row(self, X):
        pass

    def predict(self, X):
        pass

    @property
    def is_leaf(self):
        return True
