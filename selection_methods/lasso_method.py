import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
# Import base class
from .base_method import FeatureSelectionMethod

class LassoMethod(FeatureSelectionMethod):
    def __init__(self, n_features, alpha=1.0, max_iter=1000):
        super().__init__(self, n_features)
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y=None):
        metas = list(X.columns)

        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Scaling
        ('lasso', Lasso(alpha=self.alpha, max_iter=self.max_iter))  # Step 2: Lasso regression with alpha
        ])

        pipeline.fit(X, y)
        cls = pipeline.named_steps['lasso']
        
        # imp = np.sort(np.abs(cls.coef_))[-self.n_features]
        # print("imp:", imp)
        
        # Finding remained features
        sel = SelectFromModel(cls, prefit=True, max_features=self.n_features)
        # print("sel:",sel)

        mb_ids = np.array(list(range(X.shape[1])))[sel.get_support()] # Metabolites' indices
        # print("self.mb_:", self.mb_)
        self.mb_ = [metas[i] for i in mb_ids] # Metabolites' names
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