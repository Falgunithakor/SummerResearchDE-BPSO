import unittest
from src.ReadData import ReadData
from src.DEBPSO import DEBPSO

__author__ = 'FalguniT'
from sklearn import svm
from src.Population import Population
from src.SplitTypes import SplitTypes
from src.FileManager import FileManager
from src.DataManager import DataManager
from src.Velocity import Velocity


class TestDEBPSO(unittest.TestCase):

    def test_fit(self):
        #file_path = "../Dataset/00-91-Drugs-All-In-One-File.csv"
        #loaded_data = FileManager.load_file(file_path)

        read_data = ReadData()
        loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")

        data_manager = DataManager(normalizer=None)
        data_manager.set_data(loaded_data)
        data_manager.split_data_into_train_valid_test_sets()

        model = svm.SVR()

        velocity = Velocity()
        velocity_matrix = velocity.create_first_velocity()

        # define the first population
        # validation of a row generating random row for
        population = Population(velocity_matrix=velocity_matrix)
        population.create_first_population()



        debpso = DEBPSO(population.population_matrix[1])
        debpso.fit(data_manager.inputs[SplitTypes.Train], data_manager.targets[SplitTypes.Train])


    def test_transform(self):
        read_data = ReadData()
        loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")


        data_manager = DataManager(normalizer=None)
        data_manager.set_data(loaded_data)
        data_manager.split_data_into_train_valid_test_sets()

        model = svm.SVR()

        velocity = Velocity()
        velocity_matrix = velocity.create_first_velocity()

        # define the first population
        # validation of a row generating random row for
        population = Population(velocity_matrix=velocity_matrix)
        population.create_first_population()

        debpso = DEBPSO(population.population_matrix[0])
        debpso.fit(data_manager.inputs[SplitTypes.Train], data_manager.targets[SplitTypes.Train])
        data_manager.transformed_input[SplitTypes.Train] = debpso.transform(data_manager.inputs[SplitTypes.Train])
        print("Population 0 row sum ", population.population_matrix[0].sum())
        print("Selected feature descriptors",debpso.sel_descriptors_for_curr_population)
        print("Transformed array", data_manager.transformed_input[SplitTypes.Train])
