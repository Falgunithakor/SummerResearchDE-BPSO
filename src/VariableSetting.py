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
    Generation = 1
    Stop_time = 50
    Initial_alpha = 0.5
    Ending_alpha = 0.33
    Beta = 0.004
    Gamma = 3.3
    Scaling_Factor = 0.7
    Crossover_Rate = 0.7


    @staticmethod
    def set_variables(variables, no_of_drugs, no_of_descriptors):
        VariableSetting.Population_Size = int(variables[1])                    # = 50
        VariableSetting.Descriptor_Selection_Probability = variables[3]   # = 0.01
        VariableSetting.Unfit = int(variables[5])                        # = 1000,
        VariableSetting.Required_r2_Train = variables[7]                  # = 0.6,
        VariableSetting.Required_r2_Valid = variables[9]                  # = 0.5,
        VariableSetting.Required_r2_Test = variables[11]                   # = 0.5
        VariableSetting.Train_Data_Percentage = variables[13]              # = 0.7
        VariableSetting.Test_Data_Percentage   = variables[15]             # = 0.15
        VariableSetting.Generation = int(variables[17])                          # = 0.15
        VariableSetting.Stop_time = int(variables[19])                    # = 0.15
        VariableSetting.Initial_alpha= variables[21]                        # = 0.15
        VariableSetting.Ending_alpha = variables[23]                        # = 0.15
        VariableSetting.Beta = variables[25]                                # = 0.15
        VariableSetting.Gamma = variables[27]                               # = 0.15
        VariableSetting.Scaling_Factor = variables[29]                      # = 0.15
        VariableSetting.Crossover_Rate = variables[31]                      # = 0.15
        VariableSetting.No_of_Drugs = int(no_of_drugs)                         # = 91
        VariableSetting.No_of_Descriptors = int(no_of_descriptors)             # = 385
