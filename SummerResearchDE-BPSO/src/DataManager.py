import numpy as np
from SplitTypes import SplitTypes

class DataManager(object):
    def __init__(self, normalizer=None):
        self.data = None
        self.datum = {}
        self.targets = {}
        self.inputs = {}
        self.normalizer = normalizer
        '''
        self.transformed_input = {}
        self.feature_eliminator = feature_selection_algorithm

        self.num_input_columns = None
        self.num_columns = None
        '''

    def set_data(self, result):
        self.num_columns = result.shape[1]
        self.num_input_columns = self.num_columns - 1
        self.data = result
        if self.normalizer is not None:
            self.normalizer.fit(self.data[:, 0:self.num_input_columns])
            self.data[:, 0:self.num_input_columns] = self.normalizer.transform(self.data[:, 0:self.num_input_columns])

    def split_data(self, test_split, train_split):
        num_rows = self.data.shape[0]
        test_index = int(np.rint(num_rows * test_split))
        train_index = int(np.rint(num_rows * train_split)) + test_index

        self.datum = {
            SplitTypes.Train: self.data[test_index:train_index, :],
            SplitTypes.Valid: self.data[train_index:, :],
            SplitTypes.Test: self.data[0:test_index, :]
        }

        for split_type in SplitTypes.split_types_collection:
            self.inputs[split_type] = self.datum[split_type][:, 0:self.num_input_columns]
            self.targets[split_type] = self.datum[split_type][:, self.num_input_columns:self.num_columns].ravel()

    '''
    def run_default_feature_elimination(self):
        for split_type in SplitTypes.split_types_collection:
            self.inputs[split_type] = self.datum[split_type][:, 0:395]

    def run_feature_elimination(self):
        if self.feature_eliminator is None:
            self.run_default_feature_elimination()
        else:
            self.feature_eliminator.fit(self.inputs[SplitTypes.Train], self.targets[SplitTypes.Train])
            for split_type in SplitTypes.split_types_collection:
                self.transformed_input[split_type] = self.feature_eliminator.transform(self.inputs[split_type])
    '''
