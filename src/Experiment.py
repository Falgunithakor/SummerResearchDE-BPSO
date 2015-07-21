import numpy as np
from sklearn.metrics import mean_squared_error
from src.FileManager import FileManager
from src.ReadData import ReadData
from src.DEBPSO import DEBPSO
from  src.SplitTypes import SplitTypes
import matplotlib.pyplot as plt
from src.VariableSetting import VariableSetting


class Experiment(object):

    def __init__(self, data_manager, model, feature_selection_algo, output_file = "test.csv"):
        self.model = model
        self.feature_selector = feature_selection_algo
        self.data_manager = data_manager
        self.predict = {}
        self.population_predict = []
        self.r2_values = {}
        self.population_r2_values = []
        self.fitness = None
        # Appropriate for EA to obtain predictive QSAR models - as per paper
        self.gamma = 3.3

        self.alpha_scaling_factor = (VariableSetting.Initial_alpha - VariableSetting.Ending_alpha)/ VariableSetting.Generation
        self.current_alpha = VariableSetting.Initial_alpha + self.alpha_scaling_factor
        '''
        self.sum_of_squares_values = {}
        '''
        self.output_filename = output_file
    # root mean square error of train/validation, Mt, Mv, and Gamma
    def find_fitness(self):
        #print("predict value comparison", self.feature_selector.current_population_index , np.ravel(self.data_manager.targets[SplitTypes.Train]), self.predict[SplitTypes.Train] )
        NoofDescriptor = self.data_manager.transformed_input[SplitTypes.Train].shape[1]
        M_t = self.data_manager.transformed_input[SplitTypes.Train].shape[0]
        M_v = self.data_manager.transformed_input[SplitTypes.Valid].shape[0]
        RMSE_t  = np.sqrt(mean_squared_error(np.ravel(self.data_manager.targets[SplitTypes.Train]), self.predict[SplitTypes.Train]))
        RMSE_v  = np.sqrt(mean_squared_error(np.ravel(self.data_manager.targets[SplitTypes.Valid]), self.predict[SplitTypes.Valid]))

        numerator = ((M_t - NoofDescriptor -1) * (RMSE_t**2)) + (M_v * (RMSE_v ** 2))
        denominator = M_t - (self.gamma * NoofDescriptor) - 1 + M_v
        return (numerator/denominator)**2

    def fit_and_evaluate_model(self, data_inputs):
        self.model.fit(data_inputs[SplitTypes.Train], np.ravel(self.data_manager.targets[SplitTypes.Train]))
        for split_type in SplitTypes.split_types_collection:
            self.predict[split_type] = self.model.predict(data_inputs[split_type])
            self.r2_values[split_type] = self.model.score(data_inputs[split_type], (self.data_manager.targets[split_type]))
            '''
            self.sum_of_squares_values[split_type] = (
                np.sum(((self.data_manager.targets[split_type] - self.predict[split_type]) ** 2)))


        '''

        #testing area
        #r2_val = self.model.score(data_inputs[SplitTypes.Test], (self.data_manager.targets[SplitTypes.Test]))
        #sst_val = self.calculate_sumofsquaretotal((self.data_manager.targets[SplitTypes.Test]))
        #ssres = self.calculate_sumofsquareofresidual((self.data_manager.targets[SplitTypes.Test]), self.predict[SplitTypes.Test])
        #calculate_r2 = self.calculate_r2(sst_val, ssres)
        #print("X", X)
        #print("Target value", Y)
        #print("Actual Data ", data_inputs[SplitTypes.Test])
        #print("Target Data", self.data_manager.targets[SplitTypes.Test])
        #print("Predict", self.predict[SplitTypes.Test])

        #print("SST", sst_val)
        #print("SSres ", ssres)
        #print("Calculated R2", calculate_r2)


        #end - testing area


        self.population_r2_values[self.feature_selector.current_population_index][0] = self.model.score(data_inputs[SplitTypes.Train], (self.data_manager.targets[SplitTypes.Train]))
        self.population_r2_values[self.feature_selector.current_population_index][1] = self.model.score(data_inputs[SplitTypes.Valid], (self.data_manager.targets[SplitTypes.Valid]))
        self.population_r2_values[self.feature_selector.current_population_index][2] = self.model.score(data_inputs[SplitTypes.Test], (self.data_manager.targets[SplitTypes.Test]))
        self.feature_selector.fitness_matrix.append(self.find_fitness())
        #result of this function must go to a file, includes co-efficients, model name, r2 train, r2 validate, r2 test, fitness value & q2 values

    def calculate_sumofsquaretotal(self, actual_target_values):
        sst = 0
        y_bar = np.mean(actual_target_values)
        for i in range(actual_target_values.shape[0]):
            sst = sst + float((actual_target_values[i] - y_bar) ** 2)
        return sst

    def calculate_sumofsquareofresidual(self, actual_target_values, predicted_target_value):
        ssres = 0
        for i in range(actual_target_values.shape[0]):
            ssres = ssres + ((actual_target_values[i]-predicted_target_value[i])**2)
        return ssres

    def calculate_r2(self, sst, ssres):
        return 1 - (ssres / sst)

    def run_experiment(self):
        # initialize velocity and population
        # we need anther class that holds current population, velocity, current transformed matrix,
        # a reference for population row, population iteration, current generation
        # loop thru all population rows and generate fitness
        for iteration in range(1, VariableSetting.Iteration + 1):
            print("iteration loop",iteration)
            self.current_alpha = VariableSetting.Initial_alpha + self.alpha_scaling_factor
            #we can put population initialization here
            for generation in range(1, VariableSetting.Generation + 1):
                print("generation loop", generation)
                self.current_alpha = self.current_alpha - self.alpha_scaling_factor
                self.population_r2_values = np.zeros((VariableSetting.Population_Size, 3))
                self.feature_selector.fitness_matrix = []

                for population_idx in range(0, VariableSetting.Population_Size ):
                    self.feature_selector.current_population_index = population_idx
                    if self.feature_selector is None:
                        data_inputs = self.data_manager.inputs
                    else:
                        self.run_feature_selection()
                        data_inputs = self.data_manager.transformed_input
                    self.fit_and_evaluate_model(data_inputs)
                    #print("Row", population_idx, "Descriptor", self.feature_selector.sel_descriptors_for_curr_population, "Test r2 value ", self.population_r2_values[population_idx][2])
                    FileManager.write_model_in_file(self.output_filename
                                                    , self.feature_selector.sel_descriptors_for_curr_population
                                                    , self.feature_selector.fitness_matrix[population_idx]
                                                    , type(self.model)
                                                    , self.population_r2_values[population_idx][0]
                                                    , self.population_r2_values[population_idx][1]
                                                    , self.population_r2_values[population_idx][2]
                                                    )


                self.feature_selector.local_best_matrix = self.feature_selector.get_local_best_matrix()
                if generation == 1:
                    self.feature_selector.initialize_local_best_fitness_for_first_generation()
                self.feature_selector.global_best_row = self.feature_selector.get_global_row()
                #self.print_ones_in_array(self.feature_selector.global_best_row)
                self.feature_selector.find_next_velocity()
                self.feature_selector.generate_population_matrix(self.current_alpha)
                self.feature_selector.current_population_index = 0

                #print("lowest fitness index", np.min(self.feature_selector.fitness_matrix), np.argmin(self.feature_selector.fitness_matrix))
                print("Global Row fitness", self.feature_selector.global_best_row_fitness )

    #-------------------------------------------------------------------------------------------------------------------
    def print_ones_in_array(self, array):
        print("Print ones")
        for pi in range (0, 385):
            if array[pi] == 1:
                print(pi)

    def run_feature_selection(self):
        if self.feature_selector is None:
            print("none")
            self.data_manager.run_default_feature_selection()
        else:
            self.feature_selector.fit(self.data_manager.inputs[SplitTypes.Train], self.data_manager.targets[SplitTypes.Train])
            for split_type in SplitTypes.split_types_collection:
                self.data_manager.transformed_input[split_type] = self.feature_selector.transform(self.data_manager.inputs[split_type])


    '''
    def get_sum_of_squares(self, split_type):
        return self.sum_of_squares_values[split_type]
    '''

    def get_r2(self, split_type):
        return self.r2_values[split_type]

    '''
    def plot_true_vs_predicted(self, split_type):
        plt.title(SplitTypes.get_split_type_name(split_type) + " Predict Vs. Actual")
        plt.xlabel("Drug Instance")
        plt.ylabel("pIC50")
        plt.plot(self.data_manager.targets[split_type])
        plt.plot(self.predict[split_type])
        plt.show()
    '''