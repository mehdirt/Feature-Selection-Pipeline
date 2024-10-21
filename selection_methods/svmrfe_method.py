import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Import base class
from .base_method import FeatureSelectionMethod

class SVMRFEMethod(FeatureSelectionMethod):
    def __init__(self, n_features):
        super().__init__(self, n_features)

    def fit(self, X, y=None):
        metas = list(X.columns)

        # Create a pipeline with scaling and SVM-based RFE
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Scaling the data
            ('svm-rfe', RFE(SVC(kernel='linear'), n_features_to_select=self.n_features))  # SVM-RFE with linear kernel
        ])

        # Fit the pipeline to the data
        pipeline.fit(X, y)
        rfe = pipeline.named_steps['svm-rfe']
        
        # Get selected feature indices
        mb_ids = np.array(list(range(X.shape[1])))[rfe.support_]

        self.mb_ = [metas[i] for i in mb_ids] # Metabolites' names
        # print("self.mb_:", self.mb_)
        
        # Post Processing
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