import numpy as np
from src.VariableSetting import VariableSetting

__author__ = 'FalguniT'


class Population(object):
    def __init__(self, velocity_matrix):
        self.population_matrix = None
        self.velocity_matrix = velocity_matrix
        self.minimum_expected_descriptor = np.ceil(VariableSetting.Descriptor_Selection_Probability * VariableSetting.No_of_Descriptors)


    def create_first_population(self):
        '''
        We do not need to check for validity for every row as it's already done at velocity level
        Go thru every row and column and find out what the population matrix should be
        :return:
        '''
        self.population_matrix = np.zeros((VariableSetting.Population_Size, VariableSetting.No_of_Descriptors))
        #array_np = numpy.asarray(array)

        low_values_indices = self.velocity_matrix <= VariableSetting.Descriptor_Selection_Probability  # Where values are low
        self.population_matrix[low_values_indices] = 1
