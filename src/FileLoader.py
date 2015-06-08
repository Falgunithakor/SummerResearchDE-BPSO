__author__ = 'FalguniT'
import numpy as np

class FileLoader(object):
    @staticmethod
    def load_file(file_path):
        loaded_file = np.genfromtxt(file_path,delimiter=',')
        return loaded_file