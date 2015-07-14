import numpy as np
from src.VariableSetting import VariableSetting
from src.SplitTypes import SplitTypes

class DataManager(object):
    def __init__(self, normalizer=None, feature_selection_algorithm = None):
        self.data = None
        self.datum = {}
        self.targets = {}
        self.inputs = {}
        self.normalizer = normalizer
        self.feature_selector = feature_selection_algorithm
        self.transformed_input = {}
        self.num_input_columns = None
        self.num_columns = None


    def set_data(self, result):
        self.num_columns = result.shape[1]
        self.num_input_columns = self.num_columns - 1
        self.data = result


    def split_data_into_train_valid_test_sets(self):
        # no of rows = no of drugs
        test_index = int(np.rint(VariableSetting.No_of_Drugs * VariableSetting.Test_Data_Percentage))
        train_index = int(np.rint(VariableSetting.No_of_Drugs * VariableSetting.Train_Data_Percentage)) + test_index

        self.datum = {
            SplitTypes.Train: self.data[test_index:train_index, :],
            SplitTypes.Valid: self.data[train_index:, :],
            SplitTypes.Test: self.data[0:test_index, :]

        }

        for split_type in SplitTypes.split_types_collection:
            self.inputs[split_type] = self.datum[split_type][:, 0:self.num_input_columns]
            self.targets[split_type] = self.datum[split_type][:, self.num_input_columns:self.num_columns].ravel()
            if self.normalizer is not None:
                self.normalizer.fit(self.inputs[split_type][:, 0:self.num_input_columns])
                self.inputs[split_type][:, 0:self.num_input_columns] = self.normalizer.transform(self.inputs[split_type][:, 0:self.num_input_columns])



    def run_default_feature_selection(self):
        for split_type in SplitTypes.split_types_collection:
            self.inputs[split_type] = self.datum[split_type][:, 0:self.num_input_columns]


