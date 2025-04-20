"""
Feature Selection (FS) module for metabolomics data analysis.
This module provides the main FS class for orchestrating feature selection processes.
"""

import logging
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import StratifiedShuffleSplit

# Import modules
from pipeline.selection_pipeline import SelectionPipeline
from selection_methods.lasso_method import LassoMethod
from selection_methods.alasso_method import AlassoMethod
from selection_methods.elasticnet_method import ElasticNetMethod
from selection_methods.mlrrf_method import MLRRFMethod
from selection_methods.relieff_method import ReliefFMethod
from selection_methods.svmrfe_method import SVMRFEMethod
from selection_methods.boruta_method import BorutaMethod

# Configure logging
logger = logging.getLogger(__name__)

class FS:
    """
    Feature Selection class for metabolomics data analysis.
    
    This class provides methods for loading, preprocessing, and selecting features
    from metabolomics data using various feature selection methods.
    
    Attributes:
        base_panel (Set[str]): Set of base metabolites to consider
        Xtrain (pd.DataFrame): Training data
        Xtest (pd.DataFrame): Testing data
    """
    
    # Define the Base Metabolites Panel
    base_panel: Set[str] = {
        'Succinate_neg-079', 'Uridine_neg-088',
        'S-Adenosyl-methionine_pos-139',
        'N-Acetyl-D-glucosamine 6-phosphate_neg-061',
        'Serotonin_pos-142', 'Pyroglutamic acid_neg-072',
        'Neopterin_pos-117', 'Lactic acid_neg-055',
        '2-Aminooctanoic acid_pos-006', 'NMN_pos-162'
    }
    
    def __init__(self, Xtrain: pd.DataFrame, Xtest: pd.DataFrame) -> None:
        """
        Initialize the Feature Selection class.
        
        Args:
            Xtrain (pd.DataFrame): Training data
            Xtest (pd.DataFrame): Testing data
        """
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        logger.info("Initialized FS class with training and testing data")

    def create_pipeline(self) -> SelectionPipeline:
        """
        Create and configure a feature selection pipeline.
        
        Returns:
            SelectionPipeline: Configured pipeline with multiple feature selection methods
        """
        logger.info("Creating feature selection pipeline")
        pipeline = SelectionPipeline()
        
        # Add feature selection methods with their parameters
        methods = [
            LassoMethod(n_features=15, alpha=0.005, max_iter=1000),
            AlassoMethod(n_features=15, alpha=0.005, max_iter=1000),
            ElasticNetMethod(n_features=15, alpha=0.005, l1_ratio=0.5, max_iter=1000),
            MLRRFMethod(n_features=15, random_state=42),
            SVMRFEMethod(n_features=15),
            BorutaMethod(n_features=15),
            ReliefFMethod(n_features=15, n_neighbors=100)
        ]
        
        for method in methods:
            pipeline.add_method(method)
            logger.debug(f"Added {method.__class__.__name__} to pipeline")
            
        return pipeline

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess the metabolomics data.
        
        Args:
            data_path (str): Path to the Excel file containing the data
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with state column added
            
        Raises:
            FileNotFoundError: If the data file cannot be found
            ValueError: If the data format is incorrect
        """
        logger.info(f"Loading data from {data_path}")
        try:
            dframe = pd.read_excel(data_path, index_col=0)
            dframe['state'] = dframe.apply(
                lambda a: 0 if a['type'] == 'N' else 1, axis=1
            )
            logger.info(f"Successfully loaded data with {len(dframe)} samples")
            return dframe
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_batches(
        self, data_frame: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into three batches.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame containing all data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three batches of data
            
        Raises:
            ValueError: If the batch column is missing or invalid
        """
        logger.info("Splitting data into batches")
        try:
            batch1 = data_frame[data_frame['batch'] == 'batch1'].reset_index(drop=True)
            batch2 = data_frame[data_frame['batch'] == 'batch2'].reset_index(drop=True)
            batch3 = data_frame[data_frame['batch'] == 'batch3'].reset_index(drop=True)
            
            logger.info(
                f"Created batches with sizes: {len(batch1)}, {len(batch2)}, {len(batch3)}"
            )
            return batch1, batch2, batch3
        except Exception as e:
            logger.error(f"Error creating batches: {str(e)}")
            raise

    def preprocess(
        self,
        *args: pd.DataFrame,
        pipeline: SelectionPipeline = None,
        random_state: int = 42
    ) -> SelectionPipeline:
        """
        Preprocess the data and apply feature selection.
        
        Args:
            *args (pd.DataFrame): Three batches of data
            pipeline (SelectionPipeline, optional): Feature selection pipeline
            random_state (int, optional): Random seed for reproducibility
            
        Returns:
            SelectionPipeline: Pipeline with applied feature selection
            
        Raises:
            ValueError: If incorrect number of batches provided
        """
        logger.info("Starting preprocessing and feature selection")
        if len(args) != 3:
            raise ValueError("Expected exactly three batches of data")
            
        batch1, batch2, batch3 = args
        
        try:
            # Split each batch into train and test sets
            split = StratifiedShuffleSplit(
                n_splits=1, test_size=0.33, random_state=random_state
            )
            
            # Process batch1
            for train_index, test_index in split.split(batch1, batch1['type']):
                batch1_train = batch1.loc[train_index]
                batch1_test = batch1.loc[test_index]
                
            # Process batch2
            for train_index, test_index in split.split(batch2, batch2['type']):
                batch2_train = batch2.loc[train_index]
                batch2_test = batch2.loc[test_index]
                
            # Process batch3
            for train_index, test_index in split.split(batch3, batch3['type']):
                batch3_train = batch3.loc[train_index]
                batch3_test = batch3.loc[test_index]
            
            # Combine training and testing sets
            Xtrain_stratified = pd.concat(
                [batch1_train, batch2_train, batch3_train], axis=0
            ).reset_index(drop=True)
            Xtest_stratified = pd.concat(
                [batch1_test, batch2_test, batch3_test], axis=0
            ).reset_index(drop=True)
            
            # Prepare target variables
            ytrain_stratified = Xtrain_stratified['state']
            ytest_stratified = Xtest_stratified['state']
            
            # Drop unnecessary columns
            Xtrain_stratified.drop(
                ['Batch', 'type', 'batch', 'state'], axis=1, inplace=True
            )
            Xtest_stratified.drop(
                ['Batch', 'type', 'batch', 'state'], axis=1, inplace=True
            )
            
            # Convert to numpy arrays
            Xtrain = Xtrain_stratified.values
            Xtest = Xtest_stratified.values
            ytrain = np.array(ytrain_stratified, dtype=np.float32)
            ytest = np.array(ytest_stratified, dtype=np.float32)
            
            # Set metabolites list in pipeline
            pipeline.metas = list(Xtrain_stratified.columns)
            
            # Apply feature selection
            pipeline.apply(Xtrain, ytrain)
            
            logger.info("Successfully completed preprocessing and feature selection")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def extract_metas(self, metas_dict: Dict[str, Set[str]]) -> List[str]:
        """
        Extract final metabolites from the feature selection results.
        
        Args:
            metas_dict (Dict[str, Set[str]]): Dictionary of selected metabolites
                from each method
            
        Returns:
            List[str]: List of final selected metabolites
        """
        logger.info("Extracting final metabolites")
        try:
            # Get differences from base panel for each method
            differences = []
            for metas in metas_dict.values():
                differences.append(metas.difference(self.base_panel))
            
            # Find intersection of differences
            SEF = reduce(set.intersection, differences)
            # Combine with base panel
            SBP = self.base_panel.union(SEF)
            
            logger.info(f"Extracted {len(SBP)} final metabolites")
            return list(SBP)
            
        except Exception as e:
            logger.error(f"Error extracting metabolites: {str(e)}")
            raise

    @staticmethod
    def filter_data(
        dframe: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        Filter DataFrame to include only specified columns.
        
        Args:
            dframe (pd.DataFrame): Input DataFrame
            columns (List[str]): List of columns to keep
            
        Returns:
            pd.DataFrame: Filtered DataFrame
            
        Raises:
            ValueError: If any specified columns are missing
        """
        logger.info(f"Filtering data to {len(columns)} columns")
        try:
            # Check for missing columns
            missing_cols = [col for col in columns if col not in dframe.columns]
            if missing_cols:
                raise ValueError(
                    f"The following columns are not found: {missing_cols}"
                )
            
            return dframe[columns]
            
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            raise

    def select_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform the complete feature selection process.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered training and testing data
            
        Raises:
            Exception: If any step in the process fails
        """
        logger.info("Starting feature selection process")
        try:
            # Create pipeline
            pipeline = self.create_pipeline()
            
            # Load and process data
            df = self.load_data('data/discovery_set.xlsx')
            batch1, batch2, batch3 = self.create_batches(df)
            
            # Apply feature selection
            for i in range(10000):
                random_state = i
                pipeline = self.preprocess(
                    batch1, batch2, batch3,
                    pipeline=pipeline,
                    random_state=random_state
                )
            
            # Extract final metabolites
            final_metas = self.extract_metas(pipeline.method_metas)
            
            # Filter datasets
            Xtrain_final = self.filter_data(self.Xtrain, columns=final_metas)
            Xtest_final = self.filter_data(self.Xtest, columns=final_metas)
            
            logger.info(
                f"Feature selection completed. Final shape: "
                f"train={Xtrain_final.shape}, test={Xtest_final.shape}"
            )
            return Xtrain_final, Xtest_final
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise 