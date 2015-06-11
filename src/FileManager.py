import numpy as np
import time

__author__ = 'FalguniT'

class FileManager(object):
    @staticmethod
    def load_file(file_path):
        loaded_file = np.genfromtxt(file_path,delimiter=',')
        return loaded_file

    @staticmethod
    def create_output_file():
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        file_name = "../Dataset/{}.csv".format(timestamp)
        file_header = np.array([['Descriptor ID'], ['No. Descriptors'], ['Fitness'], ['Model'],['R2'], ['Q2'],
        ['R2Pred_Validation'], ['R2Pred_Test'],['SEE_Train'], ['SDEP_Validation'], ['SDEP_Test'],
        ['y_Train'], ['yHat_Train'], ['yHat_CV'], ['y_validation'], ['yHat_validation'],['y_Test'], ['yHat_Test']])
        file_header.tofile(file_name, sep=',', format='%s')
        return file_name