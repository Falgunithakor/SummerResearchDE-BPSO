from sklearn import svm
from src.DEBPSO import DEBPSO
from src.Experiment import Experiment
from src.Population import Population
from src.ReadData import ReadData

from src.SplitTypes import SplitTypes
from src.FileManager import FileManager
from src.DataManager import DataManager
from src.VariableSetting import VariableSetting
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

#data_manager.feature_selector = debpso
feature_selection_algo = None

if VariableSetting.Feature_Selection_Algorithm == 'GA' and VariableSetting.Model == 'SVM':
    #feature_selection_algo = GA()
    model = svm.SVR()
elif VariableSetting.Feature_Selection_Algorithm == 'DEBPSO' and VariableSetting.Model == 'SVM':
    feature_selection_algo = DEBPSO()
    model = svm.SVR()
#every other combination of feature and models are done same way

experiment = Experiment(data_manager, model, feature_selection_algo)
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

Velocity Alogrithm
    1) oldVeclocity = Velocity
    2) for (i=0; i<50; i++)
       {
          V1, V2, V3 = self.SelectThreeDistincVectorsFromTheOldVelocity()
          for k = 0; i< 385; i++
              V[k]' = V1[k] + F*(V2[k]- V3[k])
          for (j=0; j < 385; j++)
          {
             c = get a random number between 0 and 1
             if (c >= CR)
                 Velocity[i][j] =  V[j]'
          }
       }

   '''

'''
Lets create the second population

population = old population
for (i=0; i< 50; i++)
      for (j =0; j< 385; j++)
      {
        if alpha < V[i,j] and V[i,j] <= .5* (1+ alpha)
              X[i,j] = local_matrix[i,j]
        else i (.5) * (1+alpha) < V[i,j] and V[i, j] < (1 - beta)
             X[i, j] = globalrow[j]
        else if (1-beta) < v[i,j] and V[i,j] <= 1)
            X[i,j] = 1 - X[i,j]
     }
'''