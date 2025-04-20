"""
Main module for feature selection in metabolomics data analysis.
This module provides the entry point for the feature selection process.
"""

import logging
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from core.fs.fs import FS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main entry point for the feature selection process.
    
    This function demonstrates how to use the feature selection system:
    1. Load and prepare the data
    2. Initialize the feature selection system
    3. Perform feature selection
    4. Display and analyze the results
    """
    try:
        logger.info("Starting feature selection process")
        
        # Load the discovery set data
        discovery_data = pd.read_excel('data/discovery_set.xlsx', index_col=0)
        logger.info(f"Loaded discovery set with {len(discovery_data)} samples")
        
        # Split data into training and testing sets
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )
        
        # Prepare features and target
        X = discovery_data.drop(['Batch', 'type', 'batch', 'state'], axis=1)
        y = discovery_data['state']
        
        # Split the data
        for train_index, test_index in split.split(X, y):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        
        logger.info(
            f"Split data into train: {X_train.shape}, test: {X_test.shape}"
        )
        
        # Initialize the feature selection system
        feature_selector = FS(X_train, X_test)
        
        # Perform feature selection
        logger.info("Starting feature selection")
        X_train_selected, X_test_selected = feature_selector.select_features()
        
        # Display results
        logger.info("Feature selection completed")
        logger.info(f"Original number of features: {X_train.shape[1]}")
        logger.info(f"Selected number of features: {X_train_selected.shape[1]}")
        
        # Print selected features
        selected_features = X_train_selected.columns.tolist()
        logger.info("Selected features:")
        for feature in selected_features:
            logger.info(f"- {feature}")
            
        # Save results
        X_train_selected.to_csv('results/selected_features_train.csv')
        X_test_selected.to_csv('results/selected_features_test.csv')
        logger.info("Results saved to CSV files")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == '__main__':
    main()