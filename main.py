# Import Libraries
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# Import modules
from pipeline.selection_pipeline import SelectionPipeline
from selection_methods.lasso_method import LassoMethod
from selection_methods.alasso_method import AlassoMethod
from selection_methods.elasticnet_method import ElasticNetMethod
from selection_methods.mlrrf_method import MLRRFMethod
from selection_methods.relieff_method import ReliefFMethod
from selection_methods.svmrfe_method import SVMRFEMethod
from selection_methods.boruta_method import BorutaMethod

class FS:
    # Define the Base Metabolites Panel
    base_panel = {'Succinate_neg-079', 'Uridine_neg-088', 'S-Adenosyl-methionine_pos-139', 
              'N-Acetyl-D-glucosamine 6-phosphate_neg-061', 'Serotonin_pos-142', 
              'Pyroglutamic acid_neg-072', 'Neopterin_pos-117', 'Lactic acid_neg-055',
              '2-Aminooctanoic acid_pos-006', 'NMN_pos-162'}
    def __init__(self, Xtrain, Xtest) -> None:
        self.Xtrain = Xtrain
        self.Xtest = Xtest

    def create_pipeline(self):
        # Create a Feature Selection Pipeline
        pipeline = SelectionPipeline()

        # Add feature selection methods
        pipeline.add_method(LassoMethod(n_features=15, alpha=0.005, max_iter=1000))
        pipeline.add_method(AlassoMethod(n_features=15, alpha=0.005, max_iter=1000))
        pipeline.add_method(ElasticNetMethod(n_features=15, alpha=0.005, l1_ratio=0.5, max_iter=1000))
        pipeline.add_method(MLRRFMethod(n_features=15, random_state=42))
        pipeline.add_method(SVMRFEMethod(n_features=15))
        pipeline.add_method(BorutaMethod(n_features=15))
        pipeline.add_method(ReliefFMethod(n_features=15, n_neighbors=100))

        return pipeline

    def load_data(self, data_path):
        dframe = pd.read_excel(data_path, index_col=0)
        dframe['state'] = dframe.apply(lambda a:0 if a['type'] == 'N' else 1, axis=1)
        return dframe

    def create_batches(self, data_frame):
        batch1 = data_frame[(data_frame['batch'] == 'batch1')]
        batch1.reset_index(drop=True,inplace=True)
        batch2 = data_frame[(data_frame['batch'] == 'batch2')]
        batch2.reset_index(drop=True,inplace=True)
        batch3 = data_frame[data_frame['batch'] == 'batch3']
        batch3.reset_index(drop=True, inplace=True)

        return batch1, batch2, batch3
    
    def preprocess(self, *args, pipeline=None, random_state=42):
        batch1, batch2, batch3 = args
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=random_state)
        for train_index, test_index in split.split(batch1,batch1['type']):
            batch1_train_set = batch1.loc[train_index]
            batch1_test_set = batch1.loc[test_index] # batch1
        for train_index, test_index in split.split(batch2,batch2['type']):
            batch2_train_set = batch2.loc[train_index]
            batch2_test_set = batch2.loc[test_index] # batch2
        for train_index, test_index in split.split(batch3,batch3['type']):
            batch3_train_set = batch3.loc[train_index]
            batch3_test_set = batch3.loc[test_index] # batch3

        # 合并3个batch的training set 和testing set数据
        Xtrain_stratified = pd.concat([batch1_train_set, batch2_train_set, batch3_train_set], axis=0)
        Xtrain_stratified.reset_index(drop=True,inplace=True)
        Xtest_stratified = pd.concat([batch1_test_set, batch2_test_set, batch3_test_set], axis=0)
        Xtest_stratified.reset_index(drop=True,inplace=True)

        # 分类数据
        ytrain_stratified = Xtrain_stratified['state']
        ytest_stratified = Xtest_stratified['state']
        # 丢弃不需要的列
        Xtrain_stratified.drop(['Batch','type','batch','state'], axis=1, inplace=True)
        Xtest_stratified.drop(['Batch','type','batch','state'], axis=1, inplace=True)

        Xtrain = Xtrain_stratified
        Xtest = Xtest_stratified
        ytrain = pd.DataFrame(ytrain_stratified)
        ytest = pd.DataFrame(ytest_stratified)

        metas = list(Xtrain.columns)
        pipeline.metas = metas
        
        ytrain = np.array(ytrain.state)
        ytrain = ytrain.astype(np.float32)
        ytest = np.array(ytest.state)
        ytest = ytest.astype(np.float32)

        Xtrain = Xtrain.values
        Xtest = Xtest.values
        
        # Apply Feature Selection Pipeline on Data
        pipeline.apply(Xtrain, ytrain)

        return pipeline

    def extract_metas(self, metas_dict):
        """Return the final metabolites."""
        differences = []

        for metas in metas_dict.values():
            differences.append(metas.difference(self.base_panel))

        SEF = reduce(set.intersection, differences)
        SBP = self.base_panel.union(SEF)

        return list(SBP)

    def filter_data(dframe, columns):
        """Filter the given data according to the given columns."""
        try:
            missing_cols = [col for col in columns if col not in dframe.columns]
            if missing_cols:
                raise ValueError(f"The following columns are not found in the file: {missing_cols}")
            
            extracted_df = dframe[columns]
        
        except Exception as err:
            print("An error occurred: {err}")
        
        return extracted_df

    def select_features(self):
        # Create the pipeline
        pipeline = self.create_pipeline()
        # Load the data
        df = self.load_data('data/discovery_set.xlsx')
        # Create batches from given DataFrame
        batch1, batch2, batch3 = self.create_batches(df)
        # Get the output metabolites from each method
        for i in range(0, 10000):
            random_state = i
            pipeline = self.preprocess(batch1, batch2, batch3, pipeline=pipeline, random_state=random_state)
        
        final_metas = self.extract_metas(pipeline.method_metas)
        # Filter our datasets using extracted metabolites
        Xtrain_final = self.filter_data(self.Xtrain, columns=final_metas)
        Xtest_final = self.filter_data(self.Xtest, columns=final_metas)

        return Xtrain_final, Xtest_final
            

def main() -> None:
    pass
    # feature_selector = FS(Xtrain, Xtest)
    # filtered_Xtrain, filtered_Xtest = feature_selector.select_features()

if __name__ == '__main__':
    main()