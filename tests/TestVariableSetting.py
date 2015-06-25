import unittest
from src.FileManager import FileManager
from src.ReadData import ReadData
from src.VariableSetting import VariableSetting


class TestDEBPSO(unittest.TestCase):

    def test_verify_variable_assignment(self):
        read_data = ReadData()
        loaded_data = read_data.read_data_and_set_variable_settings("../Dataset/00-91-Drugs-All-In-One-File.csv", "../Dataset/VariableSetting.csv")

        print("Population ", VariableSetting.Population_Size)
        print("Descriptor_Selection_Probability ", VariableSetting.Descriptor_Selection_Probability)
        print("Unfit ", VariableSetting.Unfit)
        print("Required_r2_Train ", VariableSetting.Required_r2_Train)
        print("Required_r2_Valid ", VariableSetting.Required_r2_Valid)
        print("Required_r2_Test ", VariableSetting.Required_r2_Test)
        print("Generation ", VariableSetting.Generation)
        print("Stop_time ", VariableSetting.Stop_time)
        print("Initial_alpha ", VariableSetting.Initial_alpha)
        print("Ending_alpha ", VariableSetting.Ending_alpha)
        print("Beta ", VariableSetting.Beta)
        print("Gamma ", VariableSetting.Gamma)
        print("Scaling_Factor ", VariableSetting.Scaling_Factor)
        print("Crossover_Rate ", VariableSetting.Crossover_Rate)
        print("No_of_Drugs ", VariableSetting.No_of_Drugs)
        print("No_of_Descriptors ", VariableSetting.No_of_Descriptors)


