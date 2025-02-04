from micromotion_data_analysis import load_data, extract_data
import os

class ParameterConfig:
    def __init__(self):
        self.input = "data/raw_micromotion"              # File or directory to be analyzed
        self.file_to_print = "8-18_Trial18_data.txt"     # Specific trial to print data from

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
if datatype == "file":
    config.file_to_print = config.input
    with open(config.input, 'r') as file:
        tuples_list = load_data(config.input)
        voltage, height, micromotion = extract_data(tuples_list)

print(height)