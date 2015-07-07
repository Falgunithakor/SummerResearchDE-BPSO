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
        self.selective_section = int(VariableSetting.Population_Size * VariableSetting.Population_Selective_Section)
        self.old_velocity_matrix = np.zeros((VariableSetting.Population_Size, VariableSetting.No_of_Descriptors))
        self.old_population_matrix = np.zeros((VariableSetting.Population_Size, VariableSetting.No_of_Descriptors))
        self.velocity_matrix = np.zeros((VariableSetting.Population_Size, VariableSetting.No_of_Descriptors))
        self.population_matrix = np.zeros((VariableSetting.Population_Size, VariableSetting.No_of_Descriptors))
        self.local_best_matrix = np.zeros((VariableSetting.Population_Size, VariableSetting.No_of_Descriptors))
        self.global_best_row = np.zeros((VariableSetting.No_of_Descriptors))
        self.fitness_matrix = []

        self.create_first_velocity()
        self.create_first_population()
        self.local_best_matrix = self.population_matrix

    def create_first_velocity(self):
        velocity = Velocity()
        self.velocity_matrix = velocity.create_first_velocity()

    def create_first_population(self):
        population = Population(velocity_matrix=self.velocity_matrix)
        self.population_matrix = population.create_first_population()

    def get_global_row(self):
        return self.population_matrix[np.argmin(self.fitness_matrix)]


    def fit(self, X, y):
        self.current_population_row = self.population_matrix[self.current_population_index]
        self.sel_descriptors_for_curr_population = self.OnlySelectTheOnesColumns()

    def transform(self, X):
        return X.T[self.sel_descriptors_for_curr_population].T

    def OnlySelectTheOnesColumns(self):
        numOfFea = self.current_population_row.shape[0]
        xi = np.zeros(numOfFea)
        for j in range(numOfFea):
           xi[j] = self.current_population_row[j]

        xi = xi.nonzero()[0]
        xi = xi.tolist()
        return xi

    def find_next_velocity(self):
        self.old_velocity_matrix = np.copy(self.velocity_matrix)

        for row_index in range(0, VariableSetting.Population_Size):
            v_prime = self.de_algorithm()
            for col_index in range(0, VariableSetting.No_of_Descriptors):
                c = np.random.random(1)
                if c[0] >= VariableSetting.Crossover_Rate:
                    self.velocity_matrix[row_index][col_index] = v_prime[col_index]



    def de_algorithm(self):
        random_indexes = np.random.choice(VariableSetting.Population_Size, 3, replace=False)
        #print("random ", random_indexes)
        v_prime = np.zeros((VariableSetting.No_of_Descriptors))
        V1 = self.old_velocity_matrix[random_indexes[0]]
        V2 = self.old_velocity_matrix[random_indexes[1]]
        V3 = self.old_velocity_matrix[random_indexes[2]]
        for vector_index in range(0, VariableSetting.No_of_Descriptors):
            v_prime[vector_index] = V1[vector_index] +  VariableSetting.Scaling_Factor * (V2[vector_index]- V3[vector_index])

        return v_prime

    def generate_population_matrix(self, current_alpha):
        self.old_population_matrix = np.copy(self.population_matrix)

        for row_index in range(0, self.selective_section ):
            for col_index in range(0, VariableSetting.No_of_Descriptors):
                if  current_alpha< Velocity[row_index][col_index] and Velocity[row_index][col_index] <= .5* (1+ current_alpha):
                    self.population_matrix[row_index][col_index] = local_matrix[i,j]


