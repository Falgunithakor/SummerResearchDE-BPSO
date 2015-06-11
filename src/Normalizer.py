from math import sqrt
from sklearn.preprocessing import Normalizer

__author__ = 'FalguniT'

class ScikitNormalizer(object):
    def __init__(self):
        self.data_normalizer = Normalizer()

    def fit(self, data):
        self.data_normalizer.fit(data)

    def transform(self, data):
        return (self.data_normalizer.transform(data) + 1) / 2

class RescalingNormalizer(object):
    def fit(self, data):
        pass

    def transform(self, data):
        data_variance = data.var(axis = 0, ddof=1)
        data_mean = data.mean(axis = 0)
        print(data_variance)
        print(data_mean)
        #for i in range(0, data.shape[0]):
        #    data[i,:] = (data[i,:] - data_mean)/sqrt(data_variance)
        return data
    '''
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
    '''