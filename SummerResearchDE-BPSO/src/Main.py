from FileLoader import FileLoader
from DataManager import DataManager
from src.Population import Population

file_path = "../Dataset/00-91-Drugs-All-In-One-File.csv"
loaded_data = FileLoader.load_file(file_path)

data_manager = DataManager(normalizer=None)
data_manager.create_first_population(loaded_data)
data_manager.split_data_into_train_valid_test_sets(test_split=0.15, train_split=0.70)

population = Population()
population.create_first_population()
for i in range (1,50):
    print("row", i, population.population_matrix[i].sum())
