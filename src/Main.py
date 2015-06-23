from sklearn import svm
from src.Experiment import Experiment
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

file_path = "../Dataset/00-91-Drugs-All-In-One-File.csv"
loaded_data = FileManager.load_file(file_path)
#output_filename = FileManager.create_output_file()

#rescaling_normalizer = RescalingNormalizer()
#scikit_normalizer = ScikitNormalizer()
#data_manager = DataManager(normalizer=scikit_normalizer)

data_manager = DataManager(normalizer=None)
data_manager.set_data(loaded_data)
data_manager.split_data(test_split=0.15, train_split=0.70)

model = svm.SVR()

velocity = Velocity(population_size=population_size, no_of_descriptors=no_of_descriptors,  descriptor_selection_probability=descriptor_selection_probability)
velocity_matrix = velocity.create_first_velocity()


# define the first population
# validation of a row generating random row for
population = Population(population_size=population_size, no_of_descriptors=no_of_descriptors, velocity_matrix=velocity_matrix,
                        descriptor_selection_probability=descriptor_selection_probability)
population.create_first_population()

for i in range(0, population_size):
    count = 0
    for j in range(0, no_of_descriptors):
        if velocity_matrix[i,j] <= descriptor_selection_probability:
            count = count + 1
    print("Number of velocity < 0.01 for row", i , " is ", count, "Population index ", i, " row sum ", population.population_matrix[i].sum())


debpso = DEBPSO(population.population_matrix[0])
#debpso.fit(data_manager.inputs[SplitTypes.Train], data_manager.targets[SplitTypes.Train])
#data_manager.transformed_input[SplitTypes.Train] = debpso.transform(data_manager.inputs[SplitTypes.Train])


data_manager.feature_selector = debpso
experiment = Experiment(data_manager, model)
experiment.run_experiment()