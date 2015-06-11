from sklearn import svm
from src.SplitTypes import SplitTypes
from src.FileManager import FileManager
from src.DataManager import DataManager
from src.Population import Population
from src.Normalizer import *

no_of_populations = 50   # should be 50 population
no_of_descriptors = 385  # should be 385 descriptors
unfit = 1000

required_r2 = {}
required_r2[SplitTypes.Train] = .6
required_r2[SplitTypes.Valid] = .5
required_r2[SplitTypes.Test] = .5

file_path = "../Dataset/00-91-Drugs-All-In-One-File.csv"
loaded_data = FileManager.load_file(file_path)
output_filename = FileManager.create_output_file()


#rescaling_normalizer = RescalingNormalizer()
#scikit_normalizer = ScikitNormalizer()
#data_manager = DataManager(normalizer=scikit_normalizer)

data_manager = DataManager(normalizer=None)
data_manager.set_data(loaded_data)
data_manager.split_data(test_split=0.15, train_split=0.70)

model = svm.SVR()

population = Population()
population.load_data()


'''

    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR_DE_BPSO.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR_DE_BPSO.rescaleTheData(TrainX, ValidateX, TestX)

    velocity = createInitVelMat(numOfPop, numOfFea)


    def rescaleTheData(TrainX, ValidateX, TestX):

    # 1 degree of freedom means (ddof) N-1 unbiased estimation
    TrainXVar = TrainX.var(axis = 0, ddof=1)
    TrainXMean = TrainX.mean(axis = 0)

    for i in range(0, TrainX.shape[0]):
        TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
    for i in range(0, ValidateX.shape[0]):
        ValidateX[i,:] = (ValidateX[i,:] - TrainXMean)/sqrt(TrainXVar)
    for i in range(0, TestX.shape[0]):
        TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

    return TrainX, ValidateX, TestX

#------------------------------------------------------------------------------'''
