import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
# Import base class
from .base_method import FeatureSelectionMethod

class LassoMethod(FeatureSelectionMethod):
    def __init__():
        pass

    def fit(self, X, y=None):
        must = []

        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Scaling
        ('lasso', Lasso(alpha=self.args.alpha, self.args.max_iter))  # Step 2: Lasso regression with alpha
        ])

        pipeline.fit(X, y)
        cls = pipeline.named_steps['lasso']
        imp = np.sort(np.abs(cls.coef_))[-self.args.k]
        print("imp:", imp)
        sel = SelectFromModel(cls, prefit=True, max_features=self.args.k)
        print("sel:",sel)
        self.mb_ = np.array(list(range(X.shape[1])))[sel.get_support()]
        print("self.mb_:",self.mb_)
        if len(self.mb_) >= self.args.k:
            print("over max_features", self.mb_)
        if isinstance(self.mb_,list):
            self.mb_ = sorted([i for i in self.mb_ if i not in must])
            print(self.mb_)
        return self
    
    def transform(self, X):
        if len(self.mb_)!=0:
            nw_X = X[:, self.mb_]
        else: return X
        return nw_X