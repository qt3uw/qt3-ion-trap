import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-bright')

# Define file folder and parameters, 19 is great data, so is 8
path = 'C:\\Users\\Wole1\\PycharmProjects\\pythonProject\\FinalData'   # Selects the folder to pull all the data from
files = os.listdir(path)
pointstaken = 12   # The lowest n points for the minimized voltage bestfit
savemicro = True
savehist = True
savejoint = True

# Define useful lists for the loop to export values into
rf_height_vals = []
min_volt_vals = []
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

    # Isolates the lowest micromotion data points and uses a best fit curve to find the minimum between those points
    micromotion_sorted = sorted(micromotion)
    full_indices = sorted(range(len(micromotion)), key=micromotion.__getitem__)
    indices = full_indices[0:pointstaken]
    smallest_voltage = [voltage[i] for i in indices]
    smallest_micromotion = [micromotion[i] for i in indices]
    sorted_smallest_voltage = sorted(smallest_voltage)

    #plt.scatter(smallest_voltage, smallest_micromotion)
    #plt.show()

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

    rf_height_vals.append(RF_height)
    min_volt_vals.append(minvolt_raw)


    def compute_expression(y):
        # Define constants
        A = 0.000180573
        B = 0.00487685
        C = 3.26065e-8

        # Calculate the components of the expression
        numerator = -(
                A / (B + y ** 2) ** (3 / 2) + A / (y ** 2 * np.sqrt(B + y ** 2))
        )
        denominator = np.pi * (1 + C / (y ** 2 * (B + y ** 2)))

        # Final result
        result = 150 * numerator / denominator
        return result

    y_value = RF_height * 0.001  # Replace with the desired value of y
    c2mval = (9.80665 / compute_expression(y_value))
    c2mval_float = float(np.asarray(c2mval))
    charge_to_mass.append(c2mval_float)

    if file_name == '8-18_Trial19_data.txt':
        plt.figure()
        plt.scatter(voltage, micromotion)
        plt.xlabel('Voltage $(-V)$', fontsize=13)
        plt.ylabel('Micromotion $(mm)$', fontsize=13)
        plt.title('Micromotion Amplitude vs. Voltage', fontsize=15)
        plt.axvline(minvolt_raw, color='black', alpha=0.3)
        plt.errorbar(voltage, micromotion, yerr=(0.0164935065), fmt='', capsize=2, alpha=0.3, ls='none')
        plt.annotate('RF null = ' + str(int(minvolt_raw)), (int(minvolt_raw), micromotion[np.abs(voltage - minvolt_raw).argmin()]), (minvolt_raw - 40, 0.25))
        if savemicro == True:
            plt.savefig('FinalMicromotion.pdf')
        else:
            pass
        plt.show

        plt.figure()
        plt.scatter(voltage, height, color='blue')
        plt.errorbar(voltage, height, yerr=micromotion, fmt='', capsize=0, color='blue', alpha=0.3, elinewidth=4, )
        plt.xlabel('Voltage (-V)', fontsize = 13)
        plt.ylabel('Height (mm)', fontsize = 13)
        plt.annotate('RF null = (' + str(int(minvolt_raw)) + ', ' + str(np.round(RF_height, 2)[0]) + ')',
                     (minvolt_raw, RF_height), (minvolt_raw - 75, RF_height + 0.16))
        plt.axhline(RF_height, color='black', alpha=0.3)
        plt.legend(['Height', 'RF Null', 'Micromotion'])
        plt.title('Height with Micromotion Amplitude vs. Central DC Voltage', fontsize = 15)
        plt.axvline(minvolt_raw, color='black', alpha=0.3)
        if savejoint == True:
            plt.savefig('FinalJoint.pdf')
        plt.show()

plt.figure()
plt.hist(charge_to_mass, edgecolor = 'black', bins = 22, range=(-0.003,0))
plt.title('Charge to Mass Histogram', fontsize = 15)
plt.xlabel('Charge-to-Mass Ratio (C/kg)', fontsize = 13)
plt.ylabel('Frequency', fontsize = 13)
if savehist == True:
    plt.savefig('FinalHistogram.pdf')
else:
    pass
plt.show()
