__author__ = 'FalguniT'
import numpy as np
from src.VariableSetting import VariableSetting

class Velocity(object):
    def __init__(self):
        self.minimum_expected_descriptor = np.ceil(VariableSetting.Descriptor_Selection_Probability * VariableSetting.No_of_Descriptors)

    def get_valid_row(self):
        count = 0
        while count < self.minimum_expected_descriptor:
            row = np.random.random(VariableSetting.No_of_Descriptors)
            count = row[(row <= VariableSetting.Descriptor_Selection_Probability)].size
        return row

    def create_first_velocity(self):
        velocity_matrix = np.zeros((VariableSetting.Population_Size, VariableSetting.No_of_Descriptors))
        for rows in range(0, VariableSetting.Population_Size):
            valid_row = self.get_valid_row()
            velocity_matrix[rows] = valid_row
        return velocity_matrix

