"""
Feature selection pipeline module.
This module implements a pipeline for combining multiple feature selection methods.
"""

import logging
from typing import Dict, List, Set, Union
import numpy as np
from ..selection_methods.base_method import FeatureSelectionMethod

# Configure logging
logger = logging.getLogger(__name__)

class SelectionPipeline:
    """
    Pipeline for combining multiple feature selection methods.
    
    This class manages a collection of feature selection methods and applies them
    sequentially to select the most relevant features. It also handles the exclusion
    of specific metabolites based on quality criteria.
    
    Attributes:
        metas (List[str]): List of metabolite names
        methods (List[FeatureSelectionMethod]): List of feature selection methods
        method_metas (Dict[str, Set[str]]): Dictionary mapping method names to selected metabolites
        exclude_metas (List[str]): List of metabolites to exclude from selection
    """
    
    def __init__(self, metas: List[str] = None) -> None:
        """
        Initialize the feature selection pipeline.
        
        Args:
            metas (List[str], optional): List of metabolite names
        """
        self.metas = metas or []
        self.methods: List[FeatureSelectionMethod] = []
        self.method_metas: Dict[str, Set[str]] = {}
        # Metabolites excluded due to poor peak shapes in mass spectrometry
        self.exclude_metas = [
            'Glycerate-2P_Glycerate-3P_neg-006',
            'Citraconic acid_neg-025',
            'Pyridoxine_pos-137',
            'Argininosuccinic acid_pos-039'
        ]
        logger.info("Initialized SelectionPipeline")

    def add_method(self, method: FeatureSelectionMethod) -> None:
        """
        Add a feature selection method to the pipeline.
        
        Args:
            method (FeatureSelectionMethod): Feature selection method to add
        """
        self.methods.append(method)
        logger.debug(f"Added {method.__class__.__name__} to pipeline")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Set[str]]:
        """
        Apply all feature selection methods in the pipeline.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
            
        Returns:
            Dict[str, Set[str]]: Dictionary mapping method names to sets of selected metabolites
            
        Raises:
            ValueError: If no methods have been added to the pipeline
        """
        if not self.methods:
            logger.error("No feature selection methods added to pipeline")
            raise ValueError("No feature selection methods added to pipeline")
            
        logger.info(f"Applying {len(self.methods)} feature selection methods")
        
        for method in self.methods:
            method_name = method.__class__.__name__
            logger.info(f"Applying {method_name}")
            
            try:
                method.fit(X, y)
                # Get selected features and filter out excluded metabolites
                selected_indices = method.mb_
                selected_metas = {
                    self.metas[i] for i in selected_indices
                    if self.metas[i] not in self.exclude_metas
                }
                self.method_metas[method_name] = selected_metas
                logger.info(f"{method_name} selected {len(selected_metas)} metabolites")
                
            except Exception as e:
                logger.error(f"Error applying {method_name}: {str(e)}")
                raise
        
        return self.method_metas
