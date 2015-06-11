from FileLoader import FileLoader
from DataManager import DataManager
from src.Population import Population

file_path = "../Dataset/00-91-Drugs-All-In-One-File.csv"
loaded_data = FileLoader.load_file(file_path)

data_manager = DataManager(normalizer=None)
data_manager.set_data(loaded_data)
data_manager.split_data(test_split=0.15, train_split=0.70)

population = Population()
population.load_data()
for i in range (1,50):
    print("row", i, population.data[i].sum())
