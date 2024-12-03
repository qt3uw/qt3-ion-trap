import numpy as np
import matplotlib.pyplot as plt
import os
import statistics as sts
from pseudopotential import PseudopotentialPlanarTrap

# --------------------------------------- Parameters -------------------------------------------- #

FUNCTION = "SINGLE_EXTRACT"                  # "SINGLE_EXTRACT" extracts the Tuple.txt data from a single trial and outputs results
                                             # "FOLDER_EXTRACT" iterates over a folder of files and produces corresponding results files

READ_FILE = "Tuple.txt"                      # SINGLE_EXTRACT: Read file, normally Tuple.txt
SAVE_FILE = "TESTSAVEFILE.txt"               # SINGLE_EXTRACT: File save name

DATA_PATH = "data/analyzed_micromotion"      # FOLDER_EXTRACT: Folder containing raw data files
TEST_FILE = "8-18_Trial18_data.txt"          # FOLDER_EXTRACT: Single file to print analyzed data from

OUTPUT_DATA = True                           # BOTH: If True, generates and outputs new file with analyzed data
PRINT_STATS = True                           # BOTH: If True, prints statistics for the trial specified by TEST_FILE

POINTS_TAKEN = 12                            # Minimum points for voltage best fit, default 12


# -------------------------------------- Functions ----------------------------------------- #

def load_data(file_path):
    '''
    This function loads the data from the given file path.
    :param file_path:
    :return:
    '''
    tuples_list = []
    with open(file_path, 'r') as file:
        for line in file:
            tuple_str = line.strip()
            if tuple_str == "[NaN, NaN, NaN]":
                tuples_list.append([0, 0, 0])
            else:
                tuple_data = eval(tuple_str)
                tuples_list.append(tuple_data)
    return tuples_list


def extract_data(tuples_list):
    '''
    This function extracts the data from the given tuples
    :param tuples_list:
    :return:
    '''
    voltage, height, micromotion = [], [], []

    for i in range(len(tuples_list)):
        voltage.append(tuples_list[i][0])
        height.append(tuples_list[i][1])
        micromotion.append(tuples_list[i][2] / 2)
    return voltage, height, micromotion


def analyze_data(micromotion, voltage, height, file_name, testfile):
    '''
    Executes the data analysis and returns values for the RF null height, the RF null voltage, and the charge-to-mass
    :param micromotion:
    :param voltage:
    :param height:
    :param file_name:
    :param testfile:
    :return:
    '''
    micromotion_sorted = sorted(micromotion)
    full_indices = sorted(range(len(micromotion)), key=micromotion.__getitem__)
    indices = full_indices[0:POINTS_TAKEN]
    smallest_voltage = [voltage[i] for i in indices]
    smallest_micromotion = [micromotion[i] for i in indices]


    coefficients = np.polyfit(smallest_voltage, smallest_micromotion, 2)
    poly = np.poly1d(coefficients)
    x_fit = np.linspace(min(smallest_voltage), max(smallest_voltage), 100)
    y_fit = poly(x_fit)
    minvolt_raw = np.average(x_fit[np.where(y_fit == min(y_fit))])
    minvolt_int = int(minvolt_raw)

    coefficients_height = np.polyfit(voltage, height, 2)
    poly_height = np.poly1d(coefficients_height)
    y_fit_height = poly_height(np.linspace(40, 240, 201))

    index = (np.where(np.linspace(40, 240, 201) == minvolt_int))
    RF_height = y_fit_height[index]

    trap = PseudopotentialPlanarTrap()
    trap.v_dc = minvolt_raw
    c2mval = -9.80665 / trap.grad_u_dc(trap.a / 2, RF_height / 1000)

    if file_name == testfile:
        print(f'Specified Trial Q/m = {c2mval[0]}')
        print(f'Specified Trial RF Height = {RF_height[0]}')

    return RF_height, minvolt_raw, c2mval


def output_analyzed(c2mval, minvolt_raw, RF_height, file_name):
    cut_file_name = file_name.replace('.txt', '')
    with open("data/analyzed_micromotion/" + str(cut_file_name) + "_analyzed.txt", 'w') as f:
        if os.stat("data/analyzed_micromotion/" + str(cut_file_name) + "_analyzed.txt").st_size != 0:
            acknowledgement = ""
            while acknowledgement != "continue":
                acknowledgement = input('\nThe save file already contains data. Type "continue" to overwrite, otherwise use Ctrl + C to exit. ')
        f.write("[" + str(c2mval) + ", " + str(minvolt_raw) + ", " + str(RF_height[0]) + "]\n")



def print_statistics(charge_to_mass, rf_height_list):
    if PRINT_STATS:
        print('Mean Q/m =', sts.mean(charge_to_mass))
        print('Med. Q/m =', sts.median(charge_to_mass))
        print('StDev. Q/m =', sts.stdev(charge_to_mass))
        print('Mean RF Height =', sts.mean(rf_height_list))
        print('Med. RF Height =', sts.median(rf_height_list))
        print('StDev. Height =', sts.stdev(rf_height_list))


# -------------------------------------- Main Logic ----------------------------------------- #

def main():
    rf_height_vals = []
    min_volt_vals = []
    ej_volt_vals = []
    charge_to_mass = []

    try:
        files = os.listdir(DATA_PATH)
    except FileNotFoundError:
        pass

    if FUNCTION == "FOLDER_EXTRACT":
        for file_name in files:
            full_file_path = os.path.join(DATA_PATH, file_name)
            tuples_list = load_data(full_file_path)

            voltage, height, micromotion = extract_data(tuples_list)

            RF_height, minvolt_raw, c2mval = analyze_data(micromotion, voltage, height, file_name, TEST_FILE)

            rf_height_vals.append(RF_height)
            c2mval_float = float(np.asarray(c2mval[0]))
            charge_to_mass.append(c2mval_float)
            output_analyzed(c2mval_float, minvolt_raw, RF_height, file_name)

        rf_height_list = [float(val[0]) for val in rf_height_vals]
        print_statistics(charge_to_mass, rf_height_list)
    if FUNCTION == "SINGLE_EXTRACT":
        TEST_FILE = READ_FILE
        with open(READ_FILE, 'r') as file:
            tuples_list = load_data(READ_FILE)
            voltage, height, micromotion = extract_data(tuples_list)
            RF_height, minvolt_raw, c2mval = analyze_data(micromotion, voltage, height, READ_FILE, TEST_FILE)
            c2mval_float = float(np.asarray(c2mval[0]))
            print(f'Trial Q/m = {c2mval[0]}')
            print(f'Trial RF Height = {RF_height[0]}')

            if OUTPUT_DATA == True:
                output_analyzed(c2mval_float, minvolt_raw, RF_height, SAVE_FILE)
                cut_file_name = SAVE_FILE.replace('.txt', '')
                print('\nThe data has been saved to "' + "data/analyzed_micromotion/" + str(cut_file_name) + "_analyzed.txt" + '".')


if __name__ == "__main__":
    main()
