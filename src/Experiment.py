import numpy as np
from sklearn.metrics import mean_squared_error
from  src.SplitTypes import SplitTypes
import matplotlib.pyplot as plt


class Experiment(object):

    def __init__(self, data_manager, model):
        self.model = model
        self.data_manager = data_manager
        self.predict = {}
        self.r2_values = {}
        self.fitness = None
        # Appropriate for EA to obtain predictive QSAR models - as per paper
        self.gamma = 3.3

        '''
        self.sum_of_squares_values = {}
        '''
    # root mean square error of train/validation, Mt, Mv, and Gamma
    def find_fitness(self):
        NoofDescriptor = self.data_manager.transformed_input[SplitTypes.Train].shape[1]
        M_t = self.data_manager.transformed_input[SplitTypes.Train].shape[0]
        M_v = self.data_manager.transformed_input[SplitTypes.Valid].shape[0]
        RMSE_t  = np.sqrt(mean_squared_error(np.ravel(self.data_manager.targets[SplitTypes.Train]), self.predict[SplitTypes.Train]))
        RMSE_v  = np.sqrt(mean_squared_error(np.ravel(self.data_manager.targets[SplitTypes.Valid]), self.predict[SplitTypes.Valid]))

        numerator = ((M_t - NoofDescriptor -1) * (RMSE_t)**2) + (M_v * (RMSE_v ** 2))
        denominator = M_t - (self.gamma * NoofDescriptor) - 1 + M_v

        return numerator/denominator

    def fit_and_evaluate_model(self, data_inputs):
        self.model.fit(data_inputs[SplitTypes.Train], np.ravel(self.data_manager.targets[SplitTypes.Train]))
        for split_type in SplitTypes.split_types_collection:
            self.predict[split_type] = self.model.predict(data_inputs[split_type])
            self.r2_values[split_type] = self.model.score(data_inputs[split_type], (self.data_manager.targets[split_type]))
            '''
            self.sum_of_squares_values[split_type] = (
                np.sum(((self.data_manager.targets[split_type] - self.predict[split_type]) ** 2)))
            '''
        self.fitness = self.find_fitness()

    def run_experiment(self):
        # initialize velocity and population
        # we need anther class that holds current population, velocity, current transformed matrix,
        # a reference for population row, population iteration, current generation
        # loop thru all population rows and generate fitness
        if self.data_manager.feature_selector is None:
            data_inputs = self.data_manager.inputs
        else:
            self.data_manager.run_feature_selection()
            data_inputs = self.data_manager.transformed_input
        self.fit_and_evaluate_model(data_inputs)

    '''
    def get_sum_of_squares(self, split_type):
        return self.sum_of_squares_values[split_type]
    '''

    def get_r2(self, split_type):
        return self.r2_values[split_type]

    '''
    def plot_true_vs_predicted(self, split_type):
        plt.title(SplitTypes.get_split_type_name(split_type) + " Predict Vs. Actual")
        plt.xlabel("Drug Instance")
        plt.ylabel("pIC50")
        plt.plot(self.data_manager.targets[split_type])
        plt.plot(self.predict[split_type])
        plt.show()
    '''