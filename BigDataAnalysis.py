import numpy as np
import matplotlib.pyplot as plt
import os
import statistics as sts


#--------------------------------------- Define working parameters --------------------------------------------#


# Sets the plot theme
plt.style.use('seaborn-v0_8-bright')   # seaborn-v0_8-bright
plt.rcParams['font.family'] = 'Arial'
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

# Define file folder and parameters
path = 'C:\\Users\\Wole1\\PycharmProjects\\pythonProject\\AdjustedFinalData'   # Selects the folder to pull all the data from
files = os.listdir(path)
pointstaken = 12   # The lowest n points for the minimized voltage bestfit
histbins = 24
usematlabels = False
savemicro = True
savehist = True
savejoint = True

# Define useful lists for the loop to export values into
rf_height_vals = []
min_volt_vals = []
ej_volt_vals = []
charge_to_mass = []
ticker = 0

# Main loop that opens each file and analyzes each data set
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
        if len(tuples_list[i]) == 2:
            if tuples_list[i][0] != 0 and tuples_list[i][1] != 0:
                height.append(tuples_list[i][0] * amplification_factor)
                micromotion.append(tuples_list[i][1] * amplification_factor)
            if tuples_list[i][0] == 0 and tuples_list[i - 1][0] == 0:
                break
            else:
                pass
        if len(tuples_list[i]) == 3:
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

    import numpy as np

    def compute_expression(v, y):
        constant1 = 0.000180573
        constant2 = 0.00487685
        small_number = 3.26065e-8

        denom1 = (constant2 + y ** 2) ** (3 / 2)
        term1 = -constant1 / denom1
        term2 = constant1 / (y ** 2 * np.sqrt(constant2 + y ** 2))

        numerator = 2 * v * (term1 - term2)

        denom2 = 1 + (small_number / (y ** 2 * (constant2 + y ** 2)))

        result = numerator / (np.pi * denom2)

        return -result


    y_value_meters = RF_height * 0.001  # Replace with the desired value of y (m-->mm)
    c2mval = (-9.80665 / compute_expression(minvolt_raw, y_value_meters))
    c2mval_float = float(np.asarray(c2mval))
    charge_to_mass.append(c2mval_float)

    testfile = '8-18_Trial18_data.txt'

    if file_name == testfile:
        print(c2mval_float)
        print(RF_height)
        fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), height_ratios=[2, 1])
        ax1.errorbar(voltage, micromotion, yerr=(0.0164935065), color=errorcolor, fmt='', capsize=3, alpha=1, ls='none', elinewidth=3)
        ax1.scatter(voltage, micromotion, color = colorset, zorder = 3)
        if usematlabels == True:
            ax1.set_xlabel('Voltage (-V)')
            ax1.set_ylabel('Amplitude (mm)')
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
        ax2.axvline(minvolt_raw, color='black', alpha=0.6)
        plt.show
        if savejoint == True:
            plt.savefig('FinalCombo.pdf')
        else:
            pass

plt.figure()
plt.xticks(fontsize = 13.5)
plt.yticks(fontsize = 13.5)
plt.hist(charge_to_mass, edgecolor = 'black', bins=histbins, range=(-0.0015,0), color = (0.280267, 0.073417, 0.397163))
plt.show()
if usematlabels == True:
    plt.xlabel('Charge-to-Mass Ratio $\gamma$ (C/kg)')
    plt.ylabel('Number of Occurrences')
if savehist == True:
    plt.savefig('FinalHistogram.pdf')
else:
    pass

rf_height_list = []
for i in range(len(rf_height_vals)):
    rf_height_list.append(float(rf_height_vals[i]))

print('Mean = ' + str(sts.mean(charge_to_mass)))
print('Median = ' + str(sts.median(charge_to_mass)))
print('Q/m Standard Deviation = ' + str(sts.stdev(charge_to_mass)))
print('Average RF Height = ' + str(sts.mean(rf_height_list)))
print('Median RF Height = ' + str(sts.median(rf_height_list)))
print('Height Standard Deviation =' + str(sts.stdev(rf_height_list)))
print(charge_to_mass)
