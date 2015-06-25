import unittest
from src.ReadData import ReadData
from src.VariableSetting import VariableSetting
from src.DEBPSO import DEBPSO
from src.Experiment import Experiment

__author__ = 'FalguniT'
from sklearn import svm
from src.Population import Population

from src.SplitTypes import SplitTypes
from src.FileManager import FileManager
from src.DataManager import DataManager
from src.Velocity import Velocity


class TestExperiment(unittest.TestCase):
    def test_run_experiment_predict_data(self):
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

        data_manager.feature_selector = debpso
        experiment = Experiment(data_manager, model)
        experiment.run_experiment()
        expected = data_manager.transformed_input[SplitTypes.Train].shape[0]
        self.assertEqual(experiment.predict[SplitTypes.Train].shape[0], expected)

    def test_run_experiment_fitness_data_for_row_0(self):
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

        data_manager.feature_selector = debpso
        experiment = Experiment(data_manager, model)
        experiment.run_experiment()
        expected = data_manager.transformed_input[SplitTypes.Train].shape[0]
        self.assertEqual(experiment.predict[SplitTypes.Train].shape[0], expected)

    def test_run_experiment_r2_data_for_row_0(self):
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

        data_manager.feature_selector = debpso
        experiment = Experiment(data_manager, model)
        experiment.run_experiment()
        expected = data_manager.transformed_input[SplitTypes.Train].shape[0]

        print("Fitness", experiment.fitness)
        print("Train data R2", experiment.r2_values[SplitTypes.Train])
        print("Test data R2", experiment.r2_values[SplitTypes.Test])
        self.assertEqual(experiment.predict[SplitTypes.Train].shape[0], expected)

if __name__ == '__main__':
    unittest.main()
