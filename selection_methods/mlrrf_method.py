import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# Import base class
from .base_method import FeatureSelectionMethod

class MLRRFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=5, random_state=None):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.rf_model = None
        self.selected_features_ = None

    def fit(self, X, y):
        # Fit Random Forest and get feature importances
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.rf_model.fit(X, y)
        
        # Get feature importances from RandomForest
        feature_importances = self.rf_model.feature_importances_
        
        # Rank features based on importance
        ranked_features = np.argsort(feature_importances)[::-1]
        
        # Select the top `n_features_to_select` based on RandomForest importance
        top_features_rf = ranked_features[:self.n_features_to_select]
        
        # Perform Multi-Linear Regression using selected features from RF
        X_selected = X[:, top_features_rf]
        lr_model = LinearRegression()
        lr_model.fit(X_selected, y)
        
        # Get absolute values of the regression coefficients
        coefficients = np.abs(lr_model.coef_)
        
        # Rank the selected features based on the coefficients
        final_ranked_features = np.argsort(coefficients)[::-1]
        
        # Select the top features after MLR refinement
        self.selected_features_ = top_features_rf[final_ranked_features[:self.n_features_to_select]]
        
        return self

    def transform(self, X):
        # Return the selected features
        return X[:, self.selected_features_]
    
class MLRRFMethod(FeatureSelectionMethod):
    def __init__(self, n_features, random_state=42):
        super().__init__(self, n_features)
        self.random_state = random_state

    def fit(self, X, y=None):
        metas = list(X.columns)

        # Create the pipeline with StandardScaler and MLR-RF selector
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Step 1: Standardize the features
            ('mlr_rf', MLRRFSelector(n_features_to_select=self.n_features, random_state=self.random_state))  # Step 2: Apply MLR-RF
        ])
        
        # Fit the pipeline to the data
        pipeline.fit(X, y)

        # Retrieve the selected features
        mb_ids = pipeline.named_steps['mlr_rf'].selected_features_
        self.mb_ = [metas[i] for i in mb_ids] # Metabolites' names

        # Structured output
        # print(f"Selected feature indices (up to {self.n_features}): {mb_ids}")
        
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