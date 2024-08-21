import numpy as np
import matplotlib.pyplot as plt
import os
import math

trial = 9
date = "8-18"
extract_data = True
amplification_factor = 0.0162820513    #8-8=0.01935, 8-16=0.0162820513, 8-18=0.0164935065
saveplots = False
savedata = True

if extract_data == True:
    read_doc = "Tuple.txt"
else:
    read_doc = str(date) + "_Trial" + str(trial) + "_data.txt"
data_file = str(date) + '_Trial' + str(trial) + '_data.txt'
if os.path.exists(data_file) == True:
    input('This file already exists. Press Enter to continue...')
save_joint = str(date) + "_Trial" + str(trial) + "Joint.pdf"
save_micromotion = str(date) + "_Trial" + str(trial) + "Micromotion.pdf"

# Initialize an empty list to store the tuples
tuples_list = []

# Open the text file in read mode
with open(read_doc, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Strip leading/trailing whitespace and convert string to tuple
        tuple_str = line.strip()
        # Use eval() to convert the string to a tuple
        if tuple_str == "[nan, nan]":
            tuples_list.append([0,0])
        else:
            tuple_data = eval(tuple_str)
            # Append the tuple to the list
            tuples_list.append(tuple_data)

# Separates list into a height data list and a micromotion data list
height = []
micromotion = []

if tuples_list[-1][0] == 0:
    dummy = 0
else:
    dummy = 1

for i in range(len(tuples_list)):
    if len(tuples_list[i]) == 2:
        if tuples_list[i][0] != 0 and tuples_list[i][1] != 0:
            height.append(tuples_list[i][0] * amplification_factor)
            micromotion.append(tuples_list[i][1] * amplification_factor)
        if tuples_list[i][0] == 0 and tuples_list[i-1][0] == 0:
            break
        else:
            pass
    if len(tuples_list[i]) == 3:
        if tuples_list[i][1] != 0 and tuples_list[i][2] != 0:
            height.append(tuples_list[i][1])
            micromotion.append(tuples_list[i][2] / 2)
        if tuples_list[i][1] == 0 and tuples_list[i-1][1] == 0:
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

    for i in range(dummy-1):
        voltage.append(starting_voltage + (i*increment))
else:
    voltage = []
    for i in range(len(height)):
        voltage.append(tuples_list[i][0])

coefficients = np.polyfit(voltage, micromotion, 7)
poly = np.poly1d(coefficients)
x_fit = np.linspace(voltage[0], voltage[-1], 100)
y_fit = poly(x_fit)
minvolt_raw = np.average(x_fit[np.where(y_fit == min(y_fit))])

coefficients_height = np.polyfit(voltage, height, 2)
poly = np.poly1d(coefficients_height)
x_fit_height = np.linspace(voltage[0], voltage[-1], 100)
y_fit_height = poly(x_fit_height)
index = (np.where(x_fit_height == minvolt_raw))
RF_height = y_fit_height[index]

if savedata == True:
    with open(data_file, 'w') as file:
        for i in range(len(voltage)):
            file.write('[' + str(voltage[i]) + ', ' + str(round(height[i], 2)) + ', ' + str(round(micromotion[i], 2)) + ']\n' )

# Micromotion plot
plt.scatter(voltage, micromotion, color = 'green')
#plt.plot(x_fit, y_fit)
plt.xlabel('Voltage (-V)')
plt.ylabel('Micromotion (mm)')
plt.title('Micromotion Amplitude vs. Voltage')
plt.axvline(minvolt_raw, color='black', alpha = 0.3)
plt.errorbar(voltage, micromotion, yerr=0.1984375, ecolor='red', fmt='', capsize = 3, alpha = 0.3)
plt.annotate('RF null = ' + str(int(minvolt_raw)), (int(minvolt_raw), micromotion[np.abs(voltage - minvolt_raw).argmin()]), (minvolt_raw - 40, 0.35))
if saveplots == True:
    plt.savefig(save_micromotion, format='pdf')
plt.show()

# Joint plot
plt.scatter(voltage, height, color = 'blue')
plt.errorbar(voltage, height, yerr=micromotion, fmt = '', capsize = 0, color = 'blue', alpha = 0.3, elinewidth = 4, )
plt.xlabel('Voltage (-V)')
plt.ylabel('Height (mm)')
plt.annotate('RF null = (' + str(int(minvolt_raw)) + ', ' + str(np.round(RF_height, 2)[0]) + ')', (minvolt_raw, RF_height), (minvolt_raw - 65, RF_height+0.16))
plt.axhline(RF_height, color='black', alpha = 0.3)
plt.legend(['Height', 'RF Null', 'Micromotion'])
plt.title('Height with Micromotion Amplitude vs. Central DC Voltage')
plt.axvline(minvolt_raw, color='black', alpha = 0.3)
if saveplots == True:
    plt.savefig(save_joint, format='pdf')
plt.show()

