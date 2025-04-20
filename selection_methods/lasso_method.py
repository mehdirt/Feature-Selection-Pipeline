"""
Lasso-based feature selection method.
This module implements feature selection using Lasso regression.
"""

import logging
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
# Import base class
from .base_method import FeatureSelectionMethod

# Configure logging
logger = logging.getLogger(__name__)

class LassoMethod(FeatureSelectionMethod):
    """
    Feature selection using Lasso regression.
    
    This method uses Lasso regression to select features by shrinking coefficients
    of less important features to zero. It combines standardization and Lasso
    regression in a pipeline for robust feature selection.
    
    Attributes:
        n_features (int): Number of features to select
        alpha (float): Regularization strength for Lasso
        max_iter (int): Maximum number of iterations for Lasso solver
        mb_ (np.ndarray): Indices of selected features
    """
    
    def __init__(
        self,
        n_features: int,
        metas: Optional[List[str]] = None,
        alpha: float = 1.0,
        max_iter: int = 1000
    ) -> None:
        """
        Initialize the Lasso feature selection method.
        
        Args:
            n_features (int): Number of features to select
            metas (Optional[List[str]], optional): List of metabolite names
            alpha (float, optional): Regularization strength. Defaults to 1.0
            max_iter (int, optional): Maximum iterations for Lasso solver. Defaults to 1000
        """
        self.n_features = n_features
        self.alpha = alpha
        self.max_iter = max_iter
        self.metas = metas or []
        self.mb_ = np.array([])
        logger.info(
            f"Initialized LassoMethod with n_features={n_features}, "
            f"alpha={alpha}, max_iter={max_iter}"
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LassoMethod':
        """
        Fit the Lasso feature selection method to the data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y (Optional[np.ndarray], optional): Target values
            
        Returns:
            LassoMethod: The fitted feature selection method instance
        """
        logger.info("Fitting Lasso feature selection method")
        
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=self.alpha, max_iter=self.max_iter))
            ])

            pipeline.fit(X, y)
            cls = pipeline.named_steps['lasso']
            
            # Select features using the fitted Lasso model
            sel = SelectFromModel(cls, prefit=True, max_features=self.n_features)
            self.mb_ = np.array(list(range(X.shape[1])))[sel.get_support()]
            
            logger.info(f"Selected {len(self.mb_)} features using Lasso method")
            return self
            
        except Exception as e:
            logger.error(f"Error in Lasso fit: {str(e)}")
            raise

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting features using the fitted Lasso model.
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Transformed data with selected features
        """
        logger.debug("Transforming data with Lasso feature selection")
        
        if len(self.mb_) == 0:
            logger.warning("No features selected, returning original data")
            return X
            
        return X[:, self.mb_]