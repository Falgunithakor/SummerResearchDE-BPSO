import numpy as np

__author__ = 'FalguniT'


class Population(object):
    def __init__(self):
        self.data = None
    def load_data(self):
        self.data = np.zeros((50,385))
        for rows in range (1,50):
            self.data[rows] = self.validate_row_data(self.data[rows])


    def validate_row_data(self, row_data):
        row_sum = row_data.sum()
        while row_sum < 4:
            array_random = np.random.random(385)
            for inner in range (1,385):
                if array_random[inner] < 0.015:
                    row_data[inner]=1
            row_sum = row_data.sum()
        return row_data