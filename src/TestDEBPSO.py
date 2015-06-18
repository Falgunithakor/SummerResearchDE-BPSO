import unittest
from src.DEBPSO import DEBPSO

__author__ = 'FalguniT'
from sklearn import svm
from src.Population import Population

from src.SplitTypes import SplitTypes
from src.FileManager import FileManager
from src.DataManager import DataManager
from src.Velocity import Velocity


population_size = 50   # should be 50 population
no_of_descriptors = 385  # should be 385 descriptors
descriptor_selection_probability = 0.01
unfit = 1000

required_r2 = {}
required_r2[SplitTypes.Train] = .6
required_r2[SplitTypes.Valid] = .5
required_r2[SplitTypes.Test] = .5

class TestDEBPSO(unittest.TestCase):
    def test_fit(self):
        file_path = "../Dataset/00-91-Drugs-All-In-One-File.csv"
        loaded_data = FileManager.load_file(file_path)

        data_manager = DataManager(normalizer=None)
        data_manager.set_data(loaded_data)
        data_manager.split_data(test_split=0.15, train_split=0.70)

        model = svm.SVR()

        velocity = Velocity(population_size=population_size, no_of_descriptors=no_of_descriptors,  descriptor_selection_probability=descriptor_selection_probability)
        velocity_matrix = velocity.create_first_velocity()

        population = Population(population_size=population_size, no_of_descriptors=no_of_descriptors, velocity_matrix=velocity_matrix,
                        descriptor_selection_probability=descriptor_selection_probability)
        population.create_first_population()

        debpso = DEBPSO(population.population_matrix[1])
        debpso.fit(data_manager.inputs[SplitTypes.Train], data_manager.targets[SplitTypes.Train])
        print("Population 1 row sum ", population.population_matrix[1].sum())
        print("Selected feature descriptors",debpso.selected_descriptors)

