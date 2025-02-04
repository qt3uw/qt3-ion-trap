from micromotion_data_analysis import load_data, extract_data
import os
from uncertainties import ufloat

class ParameterConfig:
    def __init__(self):
        self.input = "data/raw_micromotion"              # File or directory to be analyzed
        self.file_to_print = "8-18_Trial18_data.txt"     # Specific trial to print data from
        self.dummy_height_error = 0.2                    # Dummy variable for the height value error
        self.dummy_voltage_error = 2                     # Dummy variable for the voltage value error

config = ParameterConfig()

rf_height_vals = []
charge_to_mass = []

try:
    files = os.listdir(config.input)
    datatype = "folder"
except FileNotFoundError:
    print("File could not be found. Check that you are searching in the right directory")
except NotADirectoryError:
    datatype = "file"
    pass

if datatype == "folder":
    for file_name in files:
        full_file_path = os.path.join(config.input, file_name)
        tuples_list = load_data(full_file_path)
        voltage, height, micromotion = extract_data(tuples_list)
        for i in range(len(voltage)):
            voltage[i] = ufloat(voltage[i], config.dummy_voltage_error)
            height[i] = ufloat(height[i], config.dummy_height_error)
if datatype == "file":
    config.file_to_print = config.input
    with open(config.input, 'r') as file:
        tuples_list = load_data(config.input)
        voltage, height, micromotion = extract_data(tuples_list)

print(height)