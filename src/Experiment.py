import numpy as np
from  src.SplitTypes import SplitTypes
import matplotlib.pyplot as plt


class Experiment(object):

    def __init__(self, data_manager, model):
        self.model = model
        self.data_manager = data_manager
        self.predict = {}
        '''

        self.r2_values = {}
        self.sum_of_squares_values = {}
        '''
    def find_fitness(self):
        pass

    def fit_and_evaluate_model(self, data_inputs):
        self.model.fit(data_inputs[SplitTypes.Train], np.ravel(self.data_manager.targets[SplitTypes.Train]))
        for split_type in SplitTypes.split_types_collection:
            self.predict[split_type] = self.model.predict(data_inputs[split_type])
            '''
            self.r2_values[split_type] = self.the_model.score(data_inputs[split_type],
                                                              (self.data_manager.targets[split_type]))
            self.sum_of_squares_values[split_type] = (
                np.sum(((self.data_manager.targets[split_type] - self.predict[split_type]) ** 2)))
            '''

    def run_experiment(self):
        if self.data_manager.feature_eliminator is None:
            data_inputs = self.data_manager.inputs
        else:
            self.data_manager.run_feature_elimination()
            data_inputs = self.data_manager.transformed_input
        self.fit_and_evaluate_model(data_inputs)

    '''
    def get_sum_of_squares(self, split_type):
        return self.sum_of_squares_values[split_type]
    '''
    '''
    def get_r2(self, split_type):
        return self.r2_values[split_type]
    '''
    '''
    def plot_true_vs_predicted(self, split_type):
        plt.title(SplitTypes.get_split_type_name(split_type) + " Predict Vs. Actual")
        plt.xlabel("Drug Instance")
        plt.ylabel("pIC50")
        plt.plot(self.data_manager.targets[split_type])
        plt.plot(self.predict[split_type])
        plt.show()
    '''