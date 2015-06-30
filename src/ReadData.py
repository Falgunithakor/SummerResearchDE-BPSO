import  numpy as np
from src.FileManager import FileManager
from src.VariableSetting import VariableSetting


class ReadData(object):
    def __init__(self):
        pass

    def read_data_and_set_variable_settings(self, data_file_path, variable_file_path):
        loaded_data = FileManager.load_file(data_file_path)

        no_of_drugs = loaded_data.shape[0]
        no_of_descriptors = loaded_data.shape[1] - 1    # excluding the last column that is the y axis

        variables = FileManager.load_variable_file(variable_file_path)
        VariableSetting.set_variables(variables, no_of_drugs, no_of_descriptors)
        return loaded_data