import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skrebate import ReliefF
# Import base class
from .base_method import FeatureSelectionMethod

class ReliefFMethod(FeatureSelectionMethod):
    def __init__(self, n_features, n_neighbors=100):
        super().__init__(self, n_features)
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        metas = list(X.columns)

        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling
        ('relieff', ReliefF(n_neighbors=self.n_neighbors, n_features_to_select=self.n_features, n_jobs=-1))  # ReliefF feature selection
        ])

        pipeline.fit(X, y)
        # Get the feature scores from ReliefF (the higher the score, the more relevant the feature)
        feature_scores = pipeline.named_steps['relieff'].feature_importances_
        
        # Sort features by their scores and select the top n features
        selected_indices = np.argsort(feature_scores)[-self.n_features:]
        print("Selected feature indices:", selected_indices)

        # Set the selected features to self.mb_
        mb_ids = selected_indices # Metabolites' indices
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