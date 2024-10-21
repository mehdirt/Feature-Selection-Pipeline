import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
# Import base class
from .base_method import FeatureSelectionMethod

class BorutaMethod(FeatureSelectionMethod):
    def __init__(self, n_features):
        super().__init__(self, n_features)

    def fit(self, X, y=None):
        metas = list(X.columns)

        # Apply Boruta for feature selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42, max_iter=100)

        # Standardize the features before applying Boruta
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        boruta_selector.fit(X_scaled, y)
        # Get the selected feature indices
        selected_features = np.where(boruta_selector.support_)[0]
        # print(f"Boruta selected {len(selected_features)} features.")

        # Train a RandomForest on the reduced feature set (Boruta-selected features)
        rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_final.fit(X_scaled[:, selected_features], y)

        # Get feature importances of the selected features
        importances = rf_final.feature_importances_
        
        # Rank the features based on their importance
        ranked_features = np.argsort(importances)[::-1]
        
        # Select the top most important features
        mb_ids = selected_features[ranked_features[:self.n_features]]
        # print(f"Top most important feature indices: {mb_ids}")

        self.mb_ = [metas[i] for i in mb_ids] # Metabolites' names
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
