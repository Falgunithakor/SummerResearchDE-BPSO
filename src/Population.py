import numpy as np

__author__ = 'FalguniT'


class Population(object):
    def __init__(self, population_size, no_of_descriptors, velocity_matrix, descriptor_selection_probability):
        self.data = None
        self.population_size = population_size
        self.no_of_descriptor = no_of_descriptors
        self.velocity_matrix = velocity_matrix
        # descriptor_selection_probability = lambda
        self.descriptor_selection_probability = descriptor_selection_probability
        self.minimum_expected_descriptor = np.ceil(self.descriptor_selection_probability * self.no_of_descriptor)


    def create_first_population(self):
        self.data = np.zeros((self.population_size, self.no_of_descriptor))
        for rows in range(0, self.population_size):
            population_row = self.data[rows]
            velocity_row = np.random.random(self.no_of_descriptor)
            population_row_sum = population_row.sum()
            while population_row_sum < self.minimum_expected_descriptor:
                #self.velocity_matrix[rows] = velocity_row
                #for v in range(0,self.no_of_descriptor - 1):
                #    self.velocity_matrix[rows,v] = velocity_row[v]
                for col in range(0, self.no_of_descriptor):
                    if velocity_row[col] <= self.descriptor_selection_probability:
                        #print("rows", rows,"col", col, "matrix ", self.velocity_matrix[rows, col])
                        population_row[col]= 1
                population_row_sum = population_row.sum()
        self.data[rows] = population_row
        self.velocity_matrix[rows] = velocity_row
        return self.data, self.velocity_matrix