import numpy as np

__author__ = 'FalguniT'


class Population(object):
    def __init__(self, population_size, no_of_descriptors, velocity_matrix, descriptor_selection_probability):
        self.population_matrix = None
        self.population_size = population_size
        self.no_of_descriptor = no_of_descriptors
        self.velocity_matrix = velocity_matrix
        # descriptor_selection_probability = lambda
        self.descriptor_selection_probability = descriptor_selection_probability
        self.minimum_expected_descriptor = np.ceil(self.descriptor_selection_probability * self.no_of_descriptor)


    def create_first_population(self):
        '''
        We do not need to check for validity for every row as it's already done at velocity level
        Go thru every row and column and find out what the population matrix should be
        :return:
        '''
        self.population_matrix = np.zeros((self.population_size, self.no_of_descriptor))
        #array_np = numpy.asarray(array)

        low_values_indices = self.velocity_matrix <= self.descriptor_selection_probability  # Where values are low
        self.population_matrix[low_values_indices] = 1
