# Base class for feature selection
class FeatureSelectionMethod:
    def __init__(self, n_features=5):
        self.n_features = n_features
        
    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this method")

    def transform(self, X, y):
        raise NotImplementedError("Subclasses should implement this method")

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)