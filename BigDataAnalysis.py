import numpy as np
import matplotlib.pyplot as plt
import os
import statistics as sts
from pseudopotential import PseudopotentialPlanarTrap

#--------------------------------------- Define working parameters --------------------------------------------#

'''

This code has 2 functions:

1. "Push" - Extract the raw data from the tracking code which pulls 2 valued tuples (height, micromotion) from the 
    "Tuple.txt" file, adds the voltage values, and writes the updated data into a new folder with the pixel units 
    converted to mm. The statistics for that data will be printed and graphics will be displayed.
2. "Pull" - Read a folder with existing 3 valued tuples (voltage, height, micromotion) in units of mm already, print
    the statistics, and display the relevant graphics.
    
'''

Function = "Pull"   # "Push" for option 1, "Pull" for option 2

# ------------------------------------- "Push" Parameters --------------------------------------------- #

readfile = "Tuple.txt"
savefile = "TESTSAVEFILE.txt"
amplification_factor = 0.0164935   # Uses the mm/pixel value to convert the raw pixel value to a mm measurement

# ------------------------------------- "Pull" Parameters --------------------------------------------- #

path = "Height Adjusted FInal Data"  # Selects the folder to pull all the data from
testfile = "8-18_Trial18_data.txt"
printstats = True

# ------------------------------------- General Parameters -------------------------------------------- #

pointstaken = 12   # The lowest n points for the minimized voltage bestfit
histbins = 20
showplots = False
usematlabels = False
savemicro = False
savehist = False
savejoint = False

# Sets the plot theme
plt.style.use('seaborn-v0_8-bright')   # seaborn-v0_8-bright
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.grid'] = True  # Turn on gridlines
plt.rcParams['grid.color'] = 'gray'  # Set the color of the gridlines
plt.rcParams['grid.linestyle'] = '--'  # Set the style of the gridlines (e.g., dashed)
plt.rcParams['grid.linewidth'] = 0.5  # Set the width of the gridlines
plt.rcParams['axes.labelsize'] = 18
#plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 18  # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 18  # Font size for y-axis tick labels
colorset = (0.280267, 0.073417, 0.397163)   # Purple Color
errorcolor = (0.170948, 0.694384, 0.493803)   # Green Color: (0.412913, 0.803041, 0.357269), Mint: (0.170948, 0.694384, 0.493803)

# -------------------------------------- Initialization ----------------------------------------- #

rf_height_vals = []
min_volt_vals = []
ej_volt_vals = []
charge_to_mass = []
ticker = 0

files = os.listdir(path)

def analyzedata(micromotion, voltage, height):
    # Isolates the lowest micromotion data points and uses a best fit curve to find the minimum between those points
    micromotion_sorted = sorted(micromotion)
    full_indices = sorted(range(len(micromotion)), key=micromotion.__getitem__)
    indices = full_indices[0:pointstaken]
    smallest_voltage = [voltage[i] for i in indices]
    smallest_micromotion = [micromotion[i] for i in indices]
    sorted_smallest_voltage = sorted(smallest_voltage)

    # Best fit line for the smallest micromotion data points
    coefficients = np.polyfit(smallest_voltage, smallest_micromotion, 2)
    poly = np.poly1d(coefficients)
    x_fit = np.linspace(sorted_smallest_voltage[0], sorted_smallest_voltage[-1], 100)
    y_fit = poly(x_fit)
    minvolt_raw = np.average(x_fit[np.where(y_fit == min(y_fit))])

    minvolt_int = minvolt_raw.astype(int)

    # Creates a best fit line for the height and finds its value at the minvolt_raw value (Within +-0.5V)
    x_fit_list = []
    coefficients_height = np.polyfit(voltage, height, 2)
    poly = np.poly1d(coefficients_height)
    x_fit_height = np.linspace(40, 240, 201)
    for i in range(len(x_fit_height)):
        x_fit_list.append(int(x_fit_height[i]))
    y_fit_height = poly(x_fit_height)
    index = (np.where(x_fit_list == minvolt_int))
    RF_height = y_fit_height[index]

    ej_volt_vals.append(round(tuples_list[-1][0], 2))
    rf_height_vals.append(RF_height)
    min_volt_vals.append(round(minvolt_raw, 2))

    trap = PseudopotentialPlanarTrap()
    trap.v_dc = minvolt_raw
    c2mval = -9.80665 / trap.grad_u_dc(trap.a / 2, RF_height / 1000)

    y_value_meters = RF_height * 0.001  # Replace with the desired value of y (m-->mm)
    c2mval_float = float(np.asarray(c2mval[0]))
    charge_to_mass.append(c2mval_float)

    if file_name == testfile:
        print('Specified Trial Q/m = ' + str(c2mval_float))
        print('Specified Trial RF Height = ' + str(RF_height[0]))
        fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), height_ratios=[2, 1])
        ax1.errorbar(voltage, micromotion, yerr=(0.0164935065), color=errorcolor, fmt='', capsize=3, alpha=1, ls='none', elinewidth=3)
        ax1.scatter(voltage, micromotion, color = colorset, zorder = 3)
        if usematlabels == True:
            ax1.set_xlabel('Voltage (-V)')
            ax1.set_ylabel('Amplitude (mm)')
        #ax1.set_title('Micromotion Amplitude vs. Voltage')
        ax1.axvline(minvolt_raw, color='black', alpha=0.6)
        ax1.annotate('RF null = ' + str(int(minvolt_raw)), (int(minvolt_raw), micromotion[np.abs(voltage - minvolt_raw).argmin()]), (minvolt_raw - 40, 0.25), fontsize = 18)

        ax2.scatter(voltage, height, color=colorset)
        ax2.errorbar(voltage, height, yerr=micromotion, fmt='', capsize=0, color=colorset, alpha=0.4, elinewidth=4, )
        if usematlabels == True:
            ax2.set_ylabel('Height (mm)')
        ax2.annotate('RF null = (' + str(int(minvolt_raw)) + ', ' + str(np.round(RF_height, 2)[0]) + ')',
                  (minvolt_raw, RF_height), (minvolt_raw - 40, RF_height - 0.6), fontsize = 18)
        ax2.axhline(RF_height, color='black', alpha=0.6)
        ax2.legend(['Height', 'RF Null', 'Micromotion'], fontsize = 18, loc = 'upper left')
        #ax2.set_title('Height with Micromotion Amplitude vs. Central DC Voltage')
        ax2.axvline(minvolt_raw, color='black', alpha=0.6)
        if showplots == True:
            plt.show
        if savejoint == True:
            plt.savefig('FinalCombo.pdf')
        else:
            pass

# ----------------------------------------- Main Pull Code ------------------------------------------- #

if Function == "Pull":
    for file_name in files:
        ticker = ticker + 1
        tuples_list = []
        full_file_path = os.path.join(path, file_name)
        with open(full_file_path, 'r') as file:
            # Iterate over each line in the file
            for line in file:
                # Strip leading/trailing whitespace and convert string to tuple
                tuple_str = line.strip()
                # Use eval() to convert the string to a tuple
                if tuple_str == "[nan, nan]":
                    tuples_list.append([0, 0])
                else:
                    tuple_data = eval(tuple_str)
                    # Append the tuple to the list
                    tuples_list.append(tuple_data)

        height = []
        micromotion = []

        dummy = 0
        for i in range(len(tuples_list)):
            if tuples_list[i][1] != 0 and tuples_list[i][2] != 0:
                height.append(tuples_list[i][1])
                micromotion.append(tuples_list[i][2] / 2)
            if tuples_list[i][1] == 0 and tuples_list[i - 1][1] == 0:
                break
            else:
                pass
            # Removes end zeros
            dummy = dummy + 1

        # Generates voltage vector if there isn't one (customized to data but normally 40-220)
        if len(tuples_list[0]) == 2:
            voltage = []
            increment = 5
            starting_voltage = 40
            for i in range(dummy - 1):
                voltage.append(starting_voltage + (i * increment))
        else:
            voltage = []
            for i in range(len(height)):
                voltage.append(tuples_list[i][0])

        analyzedata(micromotion, voltage, height)

        rf_height_list = []
        for i in range(len(rf_height_vals)):
            rf_height_list.append(float(rf_height_vals[i][0]))

    if printstats == True:
        print('Mean Q/m = ' + str(sts.mean(charge_to_mass)))
        print('Med. Q/m = ' + str(sts.median(charge_to_mass)))
        print('StDev. Q/m = ' + str(sts.stdev(charge_to_mass)))
        print('Mean RF Height = ' + str(sts.mean(rf_height_list)))
        print('Med. RF Height = ' + str(sts.median(rf_height_list)))
        print('StDev. Height =' + str(sts.stdev(rf_height_list)))

# --------------------------- Main Push Code ------------------------------- #

if Function == "Push":
    testfile = "Tuple.txt"
    file_name = testfile
    with open(readfile, 'r') as file:
        tuples_list = []
        # Iterate over each line in the file
        for line in file:
            # Strip leading/trailing whitespace and convert string to tuple
            tuple_str = line.strip()
            # Use eval() to convert the string to a tuple
            if tuple_str == "[nan, nan]":
                tuples_list.append([0, 0])
            else:
                tuple_data = eval(tuple_str)
                # Append the tuple to the list
                tuples_list.append(tuple_data)

    height = []
    micromotion = []
    acceptance = ""

    if tuples_list[-1][1] == 0:
        dummy = 0
    else:
        dummy = 1

    for i in range(len(tuples_list)):
        if len(tuples_list[i]) == 2:
            if tuples_list[i][0] != 0 and tuples_list[i][1] != 0:
                height.append(tuples_list[i][0] * amplification_factor)
                micromotion.append(tuples_list[i][1] * amplification_factor)
            if tuples_list[i][0] == 0 and tuples_list[i - 1][0] == 0:
                break
            else:
                pass
            # Removes end zeros
            dummy = dummy + 1
        if len(tuples_list[i]) == 3 and i == 0:
            print('\nDid you mean to use the pull function?\n')
            while acceptance != "continue":
                acceptance = input('Type "continue" to continue running the push function...')
            if tuples_list[i][1] != 0 and tuples_list[i][2] != 0:
                height.append(tuples_list[i][1])
                micromotion.append(tuples_list[i][2] / 2)
            if tuples_list[i][1] == 0 and tuples_list[i - 1][1] == 0:
                break
            else:
                pass
        else:
            if tuples_list[i][1] != 0 and tuples_list[i][2] != 0:
                height.append(tuples_list[i][1])
                micromotion.append(tuples_list[i][2] / 2)
            if tuples_list[i][1] == 0 and tuples_list[i - 1][1] == 0:
                break
            else:
                pass

    # Generates voltage vector (customized to data but normally 40-220)
    if len(tuples_list[0]) == 2:
        voltage = []
        increment = 5
        starting_voltage = 40

        for i in range(dummy - 1):
            voltage.append(starting_voltage + (i * increment))
    else:
        voltage = []
        for i in range(len(height)):
            voltage.append(tuples_list[i][0])

    if savedata == True:
        if os.stat(savefile).st_size != 0:
            acknowledgement = ""
            while acknowledgement != "continue":
                acknowledgement = input(
                    'The save file already contains data. Type "continue" to add to the existing file, otherwise cancel. ')
            print("\ncontinuing...")
        with open(savefile, 'w') as file:
            for i in range(len(voltage)):
                file.write('[' + str(voltage[i]) + ', ' + str(round(height[i], 2)) + ', ' + str(
                    round(micromotion[i], 2)) + ']\n')

    analyzedata(micromotion, voltage, height)

# --------------------------------- Data Analysis --------------------------------------- #

plt.figure()
plt.xticks(fontsize = 13.5)
plt.yticks(fontsize = 13.5)
plt.hist(charge_to_mass, edgecolor = 'black', bins=histbins, range=(-0.003,0), color = (0.280267, 0.073417, 0.397163))
#plt.title('Charge to Mass Histogram')
plt.axvline(x=-0.0025, color='black', linestyle='--', linewidth=0.5, alpha = 0.5)
plt.axvline(x=-0.0015, color='black', linestyle='--', linewidth=0.5, alpha = 0.5)
plt.axvline(x=-0.0005, color='black', linestyle='--', linewidth=0.5, alpha = 0.5)
if usematlabels == True:
    plt.xlabel('Charge-to-Mass Ratio $\gamma$ (C/kg)')
    plt.ylabel('Number of Occurrences')
if savehist == True:
    plt.savefig('FinalHistogram.pdf')
else:
    pass
if showplots == True:
    plt.show()

