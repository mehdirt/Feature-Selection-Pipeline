class FeatureSelectionMethod:
    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this method")

    def transform(self, X, y):
        raise NotImplementedError("Subclasses should implement this method")

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)