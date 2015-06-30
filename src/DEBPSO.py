from src.VariableSetting import VariableSetting
from src.Population import Population
from src.Velocity import Velocity

__author__ = 'FalguniT'
import numpy as np

class DEBPSO(object):
    def __init__(self):
        self.current_population_row = None
        self.current_population_index = 0
        self.sel_descriptors_for_curr_population = None

        self.first_velocity_matrix = {}
        self.first_population_matrix = {}
        self.first_local_matrix = {}
        self.global_row = None
        self.fitness_matrix = {}

        self.create_first_velocity()
        self.create_first_population()
        self.first_local_matrix = self.first_population_matrix

    def create_first_velocity(self):
        velocity = Velocity()
        self.first_velocity_matrix = velocity.create_first_velocity()

    def create_first_population(self):
        population = Population(velocity_matrix=self.first_velocity_matrix)
        self.first_population_matrix = population.create_first_population()

    def fit(self, X, y):
        self.current_population_row = self.first_population_matrix[self.current_population_index]
        self.sel_descriptors_for_curr_population = self.OnlySelectTheOnesColumns()


    def transform(self, X):
        self.current_population_index += 1
        return X.T[self.sel_descriptors_for_curr_population].T

    def OnlySelectTheOnesColumns(self):
        numOfFea = self.current_population_row.shape[0]
        xi = np.zeros(numOfFea)
        for j in range(numOfFea):
           xi[j] = self.current_population_row[j]

        xi = xi.nonzero()[0]
        xi = xi.tolist()
        return xi

'''
    def find_fitness_of_population(self):
        fitness = None
        for i in range(0, VariableSetting.Population_Size):
            fitness[i] = self.find_fitness(self.first_population_matrix[i])
        return  fitness

     # root mean square error of train/validation, Mt, Mv, and Gamma
    def find_fitness(self, population_array):
        NoofDescriptor = self.sel_descriptors_for_curr_population.shape[1]
        M_t = self.data_manager.transformed_input[SplitTypes.Train].shape[0]
        M_v = self.data_manager.transformed_input[SplitTypes.Valid].shape[0]
        RMSE_t  = np.sqrt(mean_squared_error(np.ravel(self.data_manager.targets[SplitTypes.Train]), self.predict[SplitTypes.Train]))
        RMSE_v  = np.sqrt(mean_squared_error(np.ravel(self.data_manager.targets[SplitTypes.Valid]), self.predict[SplitTypes.Valid]))

        numerator = ((M_t - NoofDescriptor -1) * (RMSE_t)**2) + (M_v * (RMSE_v ** 2))
        denominator = M_t - (self.gamma * NoofDescriptor) - 1 + M_v

        return numerator/denominator
'''
