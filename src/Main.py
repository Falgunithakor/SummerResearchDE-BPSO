from sklearn import svm
from src.DEBPSO import DEBPSO
from src.Experiment import Experiment
from src.Population import Population
from src.ReadData import ReadData

from src.SplitTypes import SplitTypes
from src.FileManager import FileManager
from src.DataManager import DataManager
from src.Velocity import Velocity

read_data = ReadData()
loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")

#output_filename = FileManager.create_output_file()

#rescaling_normalizer = RescalingNormalizer()
#scikit_normalizer = ScikitNormalizer()
#data_manager = DataManager(normalizer=scikit_normalizer)

data_manager = DataManager(normalizer=None)
data_manager.set_data(loaded_data)
data_manager.split_data_into_train_valid_test_sets()

model = svm.SVR()
'''
velocity = Velocity()
velocity_matrix = velocity.create_first_velocity()

# define the first population
# validation of a row generating random row for
population = Population(velocity_matrix=velocity_matrix)
population.create_first_population()

debpso = DEBPSO(population.population_matrix[0])
'''
debpso = DEBPSO()
data_manager.feature_selector = debpso
experiment = Experiment(data_manager, model)
experiment.run_experiment()


'''
For each x in feature_list  # DE-BPSO, GA, DE, PSO, ….
-	For each y in the model_list  	#MLR, PLSR, ANN, RF, SVM, …
    o	doExperiment(x, y)
    1)	create initial velocity
    2)	create the first population based on the first velocity
    3)	first_local_best_matrix = initial_population_matrix
    4)	find out the fitness of each row in the population
    5)	global_best = the best of rows in the first population

    6)	use equation 8 to find the new velocity
    7)	for each row of the velocity
        •	create three vectors from  based on DE algorithm  U = V1 + F (V2-V3)
        •	choose R = a random number between 0 and 1
            •	if R >= CR then  the row does not change row[i] = row[i]
            o	else the row becomes equal to row[i] = U
    8)	create new population based on the new velocity
    9)	revise the local best matrix if needed
    10)	revise the best global row if needed
    11)	go back to 6
'''