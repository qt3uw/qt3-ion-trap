import numpy as np
import matplotlib.pyplot as plt
import os
import statistics as sts
from pseudopotential import PseudopotentialPlanarTrap

# --------------------------------------- Constants -------------------------------------------- #

FUNCTION_PUSH = "Push"           # Set the inactive variable to "" and the active variable to it's function (Push or Pull)
FUNCTION_PULL = "False"

READ_FILE = "Tuple.txt"
SAVE_FILE = "TESTSAVEFILE.txt"
AMPLIFICATION_FACTOR = 0.0164935  # mm/pixel

DATA_PATH = "Height Adjusted FInal Data"
TEST_FILE = "8-18_Trial18_data.txt"
PRINT_STATS = True

POINTS_TAKEN = 12  # Minimum points for voltage best fit
HIST_BINS = 20
SHOW_PLOTS = False
USE_MAT_LABELS = False
SAVE_MICRO = False
SAVE_HIST = False
SAVE_JOINT = False

# Plot parameters
PLOT_STYLE = 'seaborn-v0_8-bright'
GRID_COLOR = 'gray'
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.5
LABEL_SIZE = 18

COLORS = {
    'main': (0.280267, 0.073417, 0.397163),  # Purple Color
    'error': (0.170948, 0.694384, 0.493803)  # Green Color
}


# -------------------------------------- Functions ----------------------------------------- #

def configure_plot_style():
    plt.style.use(PLOT_STYLE)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = GRID_COLOR
    plt.rcParams['grid.linestyle'] = GRID_LINESTYLE
    plt.rcParams['grid.linewidth'] = GRID_LINEWIDTH
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = LABEL_SIZE
    plt.rcParams['ytick.labelsize'] = LABEL_SIZE


def load_data(file_path):
    tuples_list = []
    with open(file_path, 'r') as file:
        for line in file:
            tuple_str = line.strip()
            if tuple_str == "[nan, nan]":
                tuples_list.append([0, 0])
            else:
                tuple_data = eval(tuple_str)
                tuples_list.append(tuple_data)
    return tuples_list


def extract_data(tuples_list):
    height = []
    micromotion = []
    for i in range(len(tuples_list)):
        if tuples_list[i][1] != 0 and len(tuples_list[i]) == 3:
            height.append(tuples_list[i][1])
            micromotion.append(tuples_list[i][2] / 2)
        if tuples_list[i][1] != 0 and len(tuples_list[i]) == 2:
            height.append(tuples_list[i][0] * AMPLIFICATION_FACTOR)
            micromotion.append((tuples_list[i][2] / 2) * AMPLIFICATION_FACTOR)
    return height, micromotion


def analyze_data(micromotion, voltage, height, file_name, testfile):
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
        plot_results(voltage, micromotion, height, minvolt_raw, RF_height)

    return RF_height, minvolt_raw, c2mval


def plot_results(voltage, micromotion, height, minvolt_raw, RF_height):
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), height_ratios=[2, 1])

    ax1.errorbar(voltage, micromotion, yerr=(0.0164935065), color=COLORS['error'], fmt='', capsize=3, alpha=1,
                 ls='none', elinewidth=3)
    ax1.scatter(voltage, micromotion, color=COLORS['main'], zorder=3)
    if USE_MAT_LABELS:
        ax1.set_xlabel('Voltage (-V)')
        ax1.set_ylabel('Amplitude (mm)')
    ax1.axvline(minvolt_raw, color='black', alpha=0.6)
    ax1.annotate(f'RF null = {int(minvolt_raw)}',
                 (int(minvolt_raw), micromotion[np.abs(voltage - minvolt_raw).argmin()]), (minvolt_raw - 40, 0.25),
                 fontsize=18)

    ax2.scatter(voltage, height, color=COLORS['main'])
    ax2.errorbar(voltage, height, yerr=micromotion, fmt='', capsize=0, color=COLORS['main'], alpha=0.4, elinewidth=4)
    if USE_MAT_LABELS:
        ax2.set_ylabel('Height (mm)')
    ax2.annotate(f'RF null = ({int(minvolt_raw)}, {np.round(RF_height, 2)[0]})',
                 (minvolt_raw, RF_height), (minvolt_raw - 40, RF_height - 0.6), fontsize=18)
    ax2.axhline(RF_height, color='black', alpha=0.6)
    ax2.legend(['Height', 'RF Null', 'Micromotion'], fontsize=18, loc='upper left')
    ax2.axvline(minvolt_raw, color='black', alpha=0.6)

    if SHOW_PLOTS:
        plt.show()
    if SAVE_JOINT:
        plt.savefig('FinalCombo.pdf')


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
    configure_plot_style()

    rf_height_vals = []
    min_volt_vals = []
    ej_volt_vals = []
    charge_to_mass = []

    files = os.listdir(DATA_PATH)

    if FUNCTION_PULL == "Pull" and FUNCTION_PUSH != "Push":
        for file_name in files:
            full_file_path = os.path.join(DATA_PATH, file_name)
            tuples_list = load_data(full_file_path)

            height, micromotion = extract_data(tuples_list)

            # Generate voltage vector
            voltage = [tuples_list[i][0] for i in range(len(height))] if len(tuples_list[0]) == 3 else list(
                range(40, 40 + len(height) * 5, 5))
            RF_height, minvolt_raw, c2mval = analyze_data(micromotion, voltage, height, file_name, TEST_FILE)

            rf_height_vals.append(RF_height)
            c2mval_float = float(np.asarray(c2mval[0]))
            charge_to_mass.append(c2mval_float)

        rf_height_list = [float(val[0]) for val in rf_height_vals]
        print_statistics(charge_to_mass, rf_height_list)
    if FUNCTION_PUSH == "Push" and FUNCTION_PULL != "Pull":
        with open(READ_FILE, 'r') as file:
            tuples_list = load_data(READ_FILE)
            height, micromotion = extract_data(tuples_list)
            voltage = [tuples_list[i][0] for i in range(len(height))] if len(tuples_list[0]) == 3 else list(
                range(40, 40 + len(height) * 5, 5))
            RF_height, minvolt_raw, c2mval = analyze_data(micromotion, voltage, height, READ_FILE, TEST_FILE)
            print(f'Trial Q/m = {c2mval[0]}')
            print(f'Trial RF Height = {RF_height[0]}')
            plot_results(voltage, micromotion, height, minvolt_raw, RF_height)
        if os.stat(SAVE_FILE).st_size != 0:
            acknowledgement = ""
            while acknowledgement != "continue":
                acknowledgement = input(
                    '\nThe save file already contains data. Type "continue" to overwrite, otherwise cancel. ')
        with open(SAVE_FILE, 'w') as file:
            for i in range(len(voltage)):
                file.write('[' + str(voltage[i]) + ', ' + str(round(height[i], 2)) + ', ' + str(
                    round(micromotion[i], 2)) + ']\n')
        print('\nThe data has been saved to "' + SAVE_FILE + '".')


if __name__ == "__main__":
    main()
