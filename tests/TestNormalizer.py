import unittest
import numpy as np
from src.ReadData import ReadData
from src.VariableSetting import VariableSetting
from src.DEBPSO import DEBPSO
from src.Experiment import Experiment
from src.Normalizer import *

__author__ = 'FalguniT'
from sklearn import svm
from src.Population import Population

from src.SplitTypes import SplitTypes
from src.FileManager import FileManager
from src.DataManager import DataManager
from src.Velocity import Velocity


class TestNormalizer(unittest.TestCase):
    def test_run_experiment_for_DEBPSO_population_With_Velocity(self):
        read_data = ReadData()
        loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")

        #output_filename = FileManager.create_output_file()

        zero_one_normalizer = ZeroOneMinMaxNormalizer()
        data_manager = DataManager(normalizer=zero_one_normalizer)


        data_manager.set_data(loaded_data)
        data_manager.split_data_into_train_valid_test_sets()

        print("Train Data", data_manager.inputs)


if __name__ == '__main__':
    unittest.main()
