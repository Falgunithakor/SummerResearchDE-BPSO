import unittest
import numpy as np
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
    '''
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
    '''


    '''
    def test_run_experiment_for_DEBPSO_population_row(self):
        read_data = ReadData()
        loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")

        #output_filename = FileManager.create_output_file()

        #rescaling_normalizer = RescalingNormalizer()
        #scikit_normalizer = ScikitNormalizer()
        #data_manager = DataManager(normalizer=scikit_normalizer)

        data_manager = DataManager(normalizer=None)
        data_manager.set_data(loaded_data)
        data_manager.split_data_into_train_valid_test_sets()

        #data_manager.feature_selector = debpso
        feature_selection_algo = None
        model = None

        if VariableSetting.Feature_Selection_Algorithm == 'GA' and VariableSetting.Model == 'SVM':
            #feature_selection_algo = GA()
            model = svm.SVR()
        elif VariableSetting.Feature_Selection_Algorithm == 'DEBPSO' and VariableSetting.Model == 'SVM':
            feature_selection_algo = DEBPSO()
            model = svm.SVR()
        #every other combination of feature and models are done same way
        experiment = Experiment(data_manager, model, feature_selection_algo)

        experiment.run_experiment()
        #expected = data_manager.transformed_input[0][SplitTypes.Train].shape[0]
        #self.assertEqual(experiment.predict[0][SplitTypes.Train].shape[0], expected)

    def test_run_experiment_for_DEBPSO_population_loop(self):
        read_data = ReadData()
        loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")

        #output_filename = FileManager.create_output_file()

        #rescaling_normalizer = RescalingNormalizer()
        #scikit_normalizer = ScikitNormalizer()
        #data_manager = DataManager(normalizer=scikit_normalizer)

        data_manager = DataManager(normalizer=None)
        data_manager.set_data(loaded_data)
        data_manager.split_data_into_train_valid_test_sets()

        #data_manager.feature_selector = debpso
        feature_selection_algo = None
        model = None

        if VariableSetting.Feature_Selection_Algorithm == 'GA' and VariableSetting.Model == 'SVM':
            #feature_selection_algo = GA()
            model = svm.SVR()
        elif VariableSetting.Feature_Selection_Algorithm == 'DEBPSO' and VariableSetting.Model == 'SVM':
            feature_selection_algo = DEBPSO()
            model = svm.SVR()
        #every other combination of feature and models are done same way
        experiment = Experiment(data_manager, model, feature_selection_algo)

        experiment.run_experiment()
        #expected = data_manager.transformed_input[0][SplitTypes.Train].shape[0]


        for i in range(0, VariableSetting.Population_Size):
            print("population row ", i, "Fitness value ",experiment.feature_selector.fitness_matrix[i],
                  "r2_values",
                  ReadData.getTwoDecPoint(experiment.population_r2_values[i][0]),
                  ReadData.getTwoDecPoint(experiment.population_r2_values[i][1]),
                  ReadData.getTwoDecPoint(experiment.population_r2_values[i][2])
               #   ReadData.getTwoDecPoint(experiment.r2_values[SplitTypes.Train][i])
            )
        print("lowest fitness index", np.min(experiment.feature_selector.fitness_matrix), np.argmin(experiment.feature_selector.fitness_matrix))
        '''

    def test_run_experiment_for_DEBPSO_population_With_Velocity(self):
        read_data = ReadData()
        loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")

        #output_filename = FileManager.create_output_file()

        #rescaling_normalizer = RescalingNormalizer()
        #scikit_normalizer = ScikitNormalizer()
        #data_manager = DataManager(normalizer=scikit_normalizer)

        data_manager = DataManager(normalizer=None)
        data_manager.set_data(loaded_data)
        data_manager.split_data_into_train_valid_test_sets()

        #data_manager.feature_selector = debpso
        feature_selection_algo = None
        model = None

        if VariableSetting.Feature_Selection_Algorithm == 'GA' and VariableSetting.Model == 'SVM':
            #feature_selection_algo = GA()
            model = svm.SVR()
        elif VariableSetting.Feature_Selection_Algorithm == 'DEBPSO' and VariableSetting.Model == 'SVM':
            feature_selection_algo = DEBPSO()
            model = svm.SVR()
        experiment = Experiment(data_manager, model, feature_selection_algo)

        experiment.run_experiment()

        print("lowest fitness index", np.min(experiment.feature_selector.fitness_matrix), np.argmin(experiment.feature_selector.fitness_matrix))

        for i in range(0, VariableSetting.Population_Size):
            print("Old Velocity", ReadData.getTwoDecPoint(experiment.feature_selector.old_velocity_matrix[i][1]), "\t"
                , ReadData.getTwoDecPoint(experiment.feature_selector.old_velocity_matrix[i][2]), "\t"
                ,ReadData.getTwoDecPoint( experiment.feature_selector.old_velocity_matrix[i][3]), "\t"
                ,ReadData.getTwoDecPoint( experiment.feature_selector.old_velocity_matrix[i][4]), "\t"
                , ReadData.getTwoDecPoint(experiment.feature_selector.old_velocity_matrix[i][5]))

            print("New Velocity",ReadData.getTwoDecPoint( experiment.feature_selector.velocity_matrix[i][1]), "\t"
                , ReadData.getTwoDecPoint(experiment.feature_selector.velocity_matrix[i][2]), "\t"
                ,ReadData.getTwoDecPoint( experiment.feature_selector.velocity_matrix[i][3]), "\t"
                , ReadData.getTwoDecPoint(experiment.feature_selector.velocity_matrix[i][4]), "\t"
                , ReadData.getTwoDecPoint(experiment.feature_selector.velocity_matrix[i][5]))
            print()


if __name__ == '__main__':
    unittest.main()
