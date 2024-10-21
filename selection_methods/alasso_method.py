import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
# Import base class
from .base_method import FeatureSelectionMethod

# WeightedLasso class for Adaptive Lasso
class WeightedLasso(Lasso):
    def __init__(self, weights, alpha=1.0, max_iter=1000, tol=0.0001, random_state=None):
        self.weights = weights
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state)

    def fit(self, X, y):
        # Scale features by their respective weights
        X_weighted = X / self.weights
        return super().fit(X_weighted, y)

class AlassoMethod(FeatureSelectionMethod):
    def __init__(self, n_features, alpha=1.0, max_iter=1000):
        super().__init__(self, n_features)
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y=None):
        metas = list(X.columns)

        # Fit an initial Lasso model to get the coefficient estimates
        initial_lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        
        # Normalize and fit the data
        X_normalized = StandardScaler().fit_transform(X)  # Use StandardScaler to normalize the data
        initial_lasso.fit(X_normalized, y)
        
        # Get initial coefficients from the first Lasso fit
        initial_coefs = initial_lasso.coef_
        
        # Define epsilon to avoid division by zero
        epsilon = 1e-4

        # Apply weights inversely proportional to the absolute value of the initial Lasso coefficients
        weights = np.where(np.abs(initial_coefs) > epsilon, 1 / np.abs(initial_coefs), 1 / epsilon)
        
        # Fit the Adaptive Lasso model
        adaptive_lasso = WeightedLasso(weights=weights, alpha=self.alpha, max_iter=self.max_iter)
        
        # Create the pipeline with StandardScaler and Adaptive Lasso
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Use StandardScaler to normalize the data
            ('adaptive_lasso', adaptive_lasso)  # Adaptive Lasso
        ])
        
        # Fit the pipeline to the data
        pipeline.fit(X, y)
        
        # Retrieve the selected features using the adaptive lasso coefficients
        imp = np.argsort(np.abs(pipeline.named_steps['adaptive_lasso'].coef_))[-self.n_features:]
        print("Selected feature indices:", imp)
        
        # Select the most important features using SelectFromModel
        sel = SelectFromModel(pipeline.named_steps['adaptive_lasso'], prefit=True, max_features=self.n_features)
        mb_ids = np.array(list(range(X.shape[1])))[sel.get_support()] # Metabolites' indices
        # print("self.mb_:", self.mb_)
        
        self.mb_ = [metas[i] for i in mb_ids] # Metabolites' names
        
        # Post-processing
        # if len(self.mb_) >= self.n_features:
        #     print("over max_features", self.mb_)
        # if isinstance(self.mb_, list):
        #     self.mb_ = sorted([i for i in self.mb_ if i not in must])
        #     print(self.mb_)
        
        return self
    
    def transform(self, X):
        if len(self.mb_)!=0:
            nw_X = X[:, self.mb_]
        else: return X
        return nw_X