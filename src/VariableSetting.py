from src.FileManager import FileManager


class VariableSetting(object):
    Population_Size = 50                        #all the following numbers are default numbers, which may not same as be used program
    Descriptor_Selection_Probability = 0.01
    Unfit = 1000
    Required_r2_Train = 0.6
    Required_r2_Valid = 0.5
    Required_r2_Test = 0.5
    Train_Data_Percentage = 0.70
    Test_Data_Percentage = 0.15
    No_of_Drugs = 91
    No_of_Descriptors = 385
    Generation = 2000
    Stop_time = 50
    Initial_alpha = 0.5
    Ending_alpha = 0.33
    Beta = 0.004
    Gamma = 3.3
    Scaling_Factor = 0.7
    Crossover_Rate = 0.7
    Feature_Selection_Algorithm = 'DEBPSO'
    Model = 'SVM'
    Iteration = 1


    @staticmethod
    def set_variables(variables, no_of_drugs, no_of_descriptors):
        VariableSetting.Population_Size = int(variables[0][1])                    # = 50
        VariableSetting.Descriptor_Selection_Probability = float(variables[1][1])   # = 0.01
        VariableSetting.Unfit = int(variables[2][1])                        # = 1000,
        VariableSetting.Required_r2_Train = float(variables[3][1])                  # = 0.6,
        VariableSetting.Required_r2_Valid = float(variables[4][1] )                 # = 0.5,
        VariableSetting.Required_r2_Test = float(variables[5][1] )                # = 0.5
        VariableSetting.Train_Data_Percentage = float(variables[6][1] )             # = 0.7
        VariableSetting.Test_Data_Percentage   = float(variables[7][1]  )         # = 0.15
        VariableSetting.Generation = int(variables[8][1])                          # = 0.15
        VariableSetting.Stop_time = int(variables[9][1])                    # = 0.15
        VariableSetting.Initial_alpha= float(variables[10][1] )                       # = 0.15
        VariableSetting.Ending_alpha =float( variables[11][1] )                      # = 0.15
        VariableSetting.Beta = float(variables[12][1])                                # = 0.15
        VariableSetting.Gamma = float(variables[13][1] )                           # = 0.15
        VariableSetting.Scaling_Factor = float(variables[14][1])                     # = 0.15
        VariableSetting.Crossover_Rate = float(variables[15][1]       )            # = 0.15
        VariableSetting.Feature_Selection_Algorithm = str(variables[16][1], encoding='ascii')      # = DEBPSO
        VariableSetting.Model = str(variables[17][1], encoding='ascii')                  # = SVM
        VariableSetting.Iteration = int(variables[18][1])                       # = 1
        VariableSetting.No_of_Drugs = int(no_of_drugs)                      # = 91
        VariableSetting.No_of_Descriptors = int(no_of_descriptors)          # = 385
