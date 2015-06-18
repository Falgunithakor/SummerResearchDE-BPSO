__author__ = 'FalguniT'
import numpy as np

class Velocity(object):
    def __init__(self, population_size, no_of_descriptors, descriptor_selection_probability):
        self.population_size = population_size
        self.no_of_descriptor = no_of_descriptors
        self.descriptor_selection_probability = descriptor_selection_probability
        self.minimum_expected_descriptor = np.ceil(self.descriptor_selection_probability * self.no_of_descriptor)

    def get_valid_row(self):
        count = 0
        while count < self.minimum_expected_descriptor:
            row = np.random.random(self.no_of_descriptor)
            count = row[(row <= self.descriptor_selection_probability)].size
        return row

    def create_first_velocity(self):
        velocity_matrix = np.zeros((self.population_size, self.no_of_descriptor))
        for rows in range(0, self.population_size):
            valid_row = self.get_valid_row()
            velocity_matrix[rows] = valid_row
        return velocity_matrix

