import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.constants import g

from pseudopotential import PseudopotentialPlanarTrap, plot_trap_escape_vary_dc

plt.style.use('seaborn-v0_8-bright')   # seaborn-v0_8-bright
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True  # Turn on gridlines
plt.rcParams['grid.color'] = 'gray'  # Set the color of the gridlines
plt.rcParams['grid.linestyle'] = '--'  # Set the style of the gridlines (e.g., dashed)
plt.rcParams['grid.linewidth'] = 0.5  # Set the width of the gridlines
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10

def get_default_trap():
    trap = PseudopotentialPlanarTrap()
    trap.v_rf = 75 * 50 * 0.5
    trap.v_dc = 209.
    trap.charge_to_mass = 2.1E-3
    return trap

def y_cuts_panel():
    trap = PseudopotentialPlanarTrap()
    trap.charge_to_mass = 2.1E-3
    fig, ax = trap.plot_y_cuts(include_gaps=True, figsize=(3.5, 3))
    fig.tight_layout()
    fig.savefig('figures/fig2-y-cuts.pdf')

def e_field_panel():
    trap = get_default_trap()
    figp, axp = trap.plot_E_field(include_gaps=True, figsize=(3.5, 3), resolution=(256, 256))
    figp.tight_layout()
    trap.v_rf = -trap.v_rf
    fign, axn = trap.plot_E_field(include_gaps=True, figsize=(3.5, 3), resolution=(256, 256))
    fign.tight_layout()
    for a in [axp, axn]:
        xticks = a.get_xticks()
        yticks = a.get_yticks()
        a.set_xticklabels([f'{tick * 1000:.0f}' for tick in xticks])
        a.set_yticklabels([f'{tick * 1000:.0f}' for tick in yticks])
        a.set_xlabel('x (mm)')
        a.set_ylabel('y (mm)')
    figp.savefig('figures/fig2-efield_positive.pdf')
    fign.savefig('figures/fig2-efield_negative.pdf')

def potential_energy_panel():
    trap = get_default_trap()
    fig, ax = trap.plot_potential_contours(include_gaps=True, figsize=(3.5, 3), max_countour_level=100, ncountours=40,
                                           resolution=(256, 256))
    for a in [ax]:
        xticks = a.get_xticks()
        yticks = a.get_yticks()
        a.set_xticklabels([f'{tick * 1000:.0f}' for tick in xticks])
        a.set_yticklabels([f'{tick * 1000:.0f}' for tick in yticks])
        a.set_xlabel('x (mm)')
        a.set_ylabel('y (mm)')
    ax.set_title(None)
    fig.tight_layout()
    fig.savefig('figures/fig2-potential_energy.pdf')

def get_data(fname='MicromotionHeightFinalData/8-18_Trial11_data.txt'):
    data_list = []

    # Read the file and process each line
    with open(fname, 'r') as file:
        for line in file:
            # Remove the brackets and whitespace, then split by commas
            line = line.strip().replace('[', '').replace(']', '')
            # Convert the split string values into floats and add them to the list
            data_list.append([float(value) for value in line.split(',')])

    # Convert the list to a NumPy array
    rawdata0= np.array(data_list)

    rawdata = np.array([[40, 0.78, 3.07], [45, 0.76, 3.09], [50, 0.73, 3.11], [55, 0.73, 3.13], [60, 0.68, 3.15],
                        [65, 0.69, 3.2], [70, 0.66, 3.2], [75, 0.65, 3.22], [80, 0.61, 3.26], [85, 0.59, 3.28],
                        [90, 0.57, 3.33], [95, 0.54, 3.35], [100, 0.52, 3.37], [105, 0.51, 3.38], [110, 0.5, 3.42],
                        [115, 0.47, 3.45], [120, 0.45, 3.48], [125, 0.42, 3.53], [130, 0.4, 3.56], [135, 0.39, 3.58],
                        [140, 0.34, 3.62], [145, 0.35, 3.65], [150, 0.32, 3.67], [155, 0.3, 3.72], [160, 0.26, 3.76],
                        [165, 0.26, 3.79], [170, 0.21, 3.83], [175, 0.2, 3.87], [180, 0.19, 3.9], [185, 0.19, 3.95],
                        [190, 0.17, 3.99], [195, 0.19, 4.02], [200, 0.23, 4.06], [205, 0.29, 4.11], [210, 0.38, 4.25]])
    rawdata = rawdata[:-1]
    dc_voltages = rawdata[:, 0]
    y_spread = rawdata[:, 1]
    y0 = rawdata[:, 2]
    meas_min = rawdata[np.argmin(rawdata[:, 1])]
    #Rought guess based on chart. Need more precise measurement from Cole.
    return dc_voltages, y0 * 1.E-3, y_spread * 1.E-3, meas_min

def plot_height_fit(include_gaps=True, figsize=(3.5, 3)):
    trap = get_default_trap()
    parameters = ['charge_to_mass']
    bounds = [(1.E-4, 1.E-2)]

    dc_voltages, y0, yspread, meas_min = get_data()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    guesses = [trap.__dict__[param] for param in parameters]

    def merit_func(args):
        for i, key in enumerate(parameters):
            trap.__dict__[key] = args[i]
        y0_model = trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps)
        return np.sum((y0 - y0_model) ** 2)

    res = minimize(merit_func, guesses, bounds=bounds)

    for i, param in enumerate(parameters):
        print(f'{param}: {res.x[i]}')
        trap.__dict__[param] = res.x[i]

    model_voltages = np.linspace(np.min(dc_voltages), np.max(dc_voltages), num=100)
    y0_model = trap.get_height_versus_dc_voltages(model_voltages, include_gaps=include_gaps)
    ax.plot(model_voltages, y0_model * 1.E3, color='k', label='best fit')
    trap.v_dc = meas_min[0]
    rf_null_meas = trap.grad_u_dc(trap.a / 2, meas_min[2]*10**-3, include_gaps=include_gaps)
    trap.charge_to_mass = -g/rf_null_meas
    print("c_t_m: " + str(trap.charge_to_mass))
    y0_meas = trap.get_height_versus_dc_voltages(model_voltages, include_gaps=include_gaps)
    ax.plot(dc_voltages, y0 * 1.E3, marker='o', linestyle='None')
    plt.errorbar(dc_voltages, y0 * 1.E3, yerr=0.0164, fmt='none', ls='none', capsize=5)
    ax.plot(model_voltages, y0_meas * 1.E3, color='k', linestyle='--', label='rf null')
    ax.set_xlabel('DC electrode voltage (V)')
    ax.set_ylabel('Ion height (mm)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/fig2-height_fit.pdf')
    # fig.suptitle(f'include_gaps={include_gaps}')
    return trap

def plot_escape():
    trap = get_default_trap()
    fig, ax = plot_trap_escape_vary_dc(trap, dc_values=np.linspace(0., 300., num=11), include_gaps=True)
    fig.tight_layout()
    ax.set_ylabel('potential energy / charge (J/C)')
    ax.set_title(None)
    fig.savefig('figures/fig2-trap_escape.pdf')




if __name__ == "__main__":
    # y_cuts_panel()
    # e_field_panel()
    # potential_energy_panel()
    # plot_escape()
    plot_height_fit(figsize=(2.5, 3.))
    plt.show()