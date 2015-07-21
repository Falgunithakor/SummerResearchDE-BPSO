import numpy as np
import time

__author__ = 'FalguniT'

class FileManager(object):
    @staticmethod
    def load_file(file_path):
        loaded_file = np.genfromtxt(file_path,delimiter=',')
        return loaded_file

    @staticmethod
    def load_variable_file(file_path):
         loaded_file = np.loadtxt(file_path, delimiter=',',  dtype={'names': ('Name', 'Value'),  'formats': ('S1','S10')})
         return loaded_file

    @staticmethod
    def create_output_file(feature_selection_alogorithm ='DEBPSO', model = 'SVR'):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        file_name = "../Dataset/{}-{}-{}.csv".format(feature_selection_alogorithm, model,timestamp)
        with open(file_name,"a") as f_handle:
            f_handle.write('Descriptor ID, Fitness, Model, R2_Train, R2Pred_Validation, R2Pred_Test\n')
        #file_header.tofile(file_name, sep=',', format='%s', newline='\n')
        #np.savetxt(file_name,file_header,fmt='%s', delimiter=',', newline='\n')
        return file_name

    @staticmethod
    def write_model_in_file(file_name, descriptor_ids,  fitness, model, r2_train, r2pred_validation, r2pred_test):
        with open(file_name,"a") as f_handle:
            f_handle.write( str(descriptor_ids).replace(',','-') + ','+ str(fitness) + ','+ str(model)  + ','+ str(r2_train) + ','+ str(r2pred_validation) + ','+ str(r2pred_test)+ '\n')



