"""
Base class for feature selection methods.
This module provides the abstract base class for all feature selection methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureSelectionMethod(ABC):
    """
    Abstract base class for feature selection methods.
    
    This class defines the interface that all feature selection methods must implement.
    Subclasses should implement the fit and transform methods to provide specific
    feature selection functionality.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> 'FeatureSelectionMethod':
        """
        Fit the feature selection method to the data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y (Union[np.ndarray, None], optional): Target values of shape (n_samples,)
            
        Returns:
            FeatureSelectionMethod: The fitted feature selection method instance
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        logger.debug("Fitting feature selection method")
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def transform(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> np.ndarray:
        """
        Transform the data by selecting features.
        
        Args:
            X (np.ndarray): Data to transform of shape (n_samples, n_features)
            y (Union[np.ndarray, None], optional): Target values of shape (n_samples,)
            
        Returns:
            np.ndarray: Transformed data with selected features
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        logger.debug("Transforming data with feature selection method")
        raise NotImplementedError("Subclasses should implement this method")

    def fit_transform(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> np.ndarray:
        """
        Fit the feature selection method and transform the data in one step.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y (Union[np.ndarray, None], optional): Target values of shape (n_samples,)
            
        Returns:
            np.ndarray: Transformed data with selected features
        """
        logger.info("Performing fit_transform operation")
        self.fit(X, y)
        return self.transform(X)