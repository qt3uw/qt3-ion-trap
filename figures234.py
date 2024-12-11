import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.constants import g
import os

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
plt.rcParams['axes.titlesize'] = 12
COLORS = {
        'main': (0.280267, 0.073417, 0.397163),  # Purple Color
        'error': (0.170948, 0.694384, 0.493803)  # Green Color
}


class FigureParameterConfig:
    def __init__(self):
        self.save_fig = True                                                    # Saves figure to directory specified by self.save_path
        self.pixel_to_mm = 0.0164935065                                         # Pixel to mm conversion from calibration. Only for plotting error bars, okay to set to zero if trials vary
        self.graph_file_name = 'acquisition/ExampleMicromotion_data.txt'        # File to plot height & micromotion vs. voltage graphs for
        self.hist_folder_name = 'data/analyzed_micromotion'                     # Folder to extract charge-to-mass values from and graph the histogram
        self.save_path = ["figures/figure_" + str(i) + "/" for i in range(2, 5)]                                                # Path for exported figures


def get_default_config():
    return FigureParameterConfig()

  
def get_default_trap():
    """
    Creates a and returns a trap object
    :return: A trap object from the PseudopotentialPlanarTrap class
    """
    trap = PseudopotentialPlanarTrap()
    trap.v_rf = -75 * 50 * 0.5
    trap.charge_to_mass = -1.077E-3
    return trap


def y_cuts_panel():
    """
    Plots and saves the potential energy divided by charge of the various relevant scalar fields
    """
    config = get_default_config()
    trap = get_default_trap()
    trap.v_dc = -80.
    fig, ax = trap.plot_y_cuts(include_gaps=True, figsize=(3.5, 3))
    fig.tight_layout()
    os.makedirs(config.save_path[0], exist_ok =True)
    fig.savefig(config.save_path[0] +"fig2-y-cuts.pdf")


def e_field_panel():
    """
    Plots and saves the electric field of the planar trap.
    """
    config = get_default_config()
    trap = get_default_trap()
    figp, axp = trap.plot_E_field(include_gaps=True, x_range=(-trap.c, trap.a + trap.b), normalized = False,
                                  resolution=(256, 256), figsize=(6, 3.5))
    os.makedirs(config.save_path[0], exist_ok =True)
    figp.savefig(config.save_path[0] +"fig2-efield.pdf")


def potential_energy_panel():
    """
    Plots and saves the pseudopotential scalar field and equipotential contour lines.
    """
    trap = get_default_trap()
    config = get_default_config()
    fig, ax = trap.plot_rf_potential_contours(include_gaps=True, figsize=(4.1, 3), x_range=(-trap.c, trap.a + trap.b),
                                              min_contour_level=-20, ncountours=41, resolution=(256, 256))
    for a in [ax]:
        xticks = a.get_xticks()
        yticks = a.get_yticks()
        a.set_xticklabels([f'{tick * 1000:.0f}' for tick in xticks])
        a.set_yticklabels([f'{tick * 1000:.0f}' for tick in yticks])
        a.set_xlabel('x (mm)')
        a.set_ylabel('y (mm)')
    ax.set_title(None)
    fig.tight_layout()
    os.makedirs(config.save_path[0], exist_ok =True)
    fig.savefig(config.save_path[0]+"fig2-potential_energy.pdf")

    
def get_data(filename = None, config = get_default_config()):
    """
    Reads and sorts experimental data from a text file.
    :param filename: File to extract data from. Only necessary for folder iteration in which the file is not graph_file_name
    :return: If filename is specificed, returns the data points, where each point has the following form-
             (DC voltage, centroid, micromotion amplitude,
             voltage when micromotion is minimized,
             centroid when micromotion is minimized,
             minimum micromotion amolitude)
             Otherwise, returns only the charge-to-mass value
    """
    data_list = []
    if filename == None:
        datafile = config.graph_file_name
    else: 
        datafile = filename
    basefilename = os.path.basename(datafile)
    cut_basefilename = basefilename.replace('.txt', '')
    if filename == None:
        analyzedfilename = 'data/analyzed_micromotion/' + str(cut_basefilename) + '_analyzed.txt'
        with open(datafile, 'r') as file:
            for line in file:
                line = line.strip().replace('[', '').replace(']', '')
                data_list.append([float(value) for value in line.split(',')])
            rawdata = np.array(data_list)
            rawdata = rawdata[:-1]
            dc_voltages = rawdata[:, 0]
            y_spread = rawdata[:, 2]
            y0 = rawdata[:, 1]
            v_min, y_min, micro_min = rawdata[np.argmin(rawdata[:, 2])]
        with open(analyzedfilename) as file:
            for line in file:
                line = line.strip()
                analyzed_data = eval(line)
                c2m, null_volt, null_height = analyzed_data[0], analyzed_data[1], analyzed_data[2]
        return -dc_voltages, y0 * 1.E-3, y_spread * 1.E-3, v_min, y_min * 1.E-3, micro_min * 1.E-3, c2m, null_volt, null_height
    else:
        analyzedfilename = 'data/analyzed_micromotion/' + str(cut_basefilename) + '.txt'
    with open(analyzedfilename) as file:
        for line in file:
            line = line.strip()
            analyzed_data = eval(line)
            c2m, null_volt, null_height = analyzed_data[0], analyzed_data[1], analyzed_data[2]
    return c2m
    

def plot_height_fit(include_gaps=True, figsize=(3.5, 3)):
    """
    Plots and saves experimental ion height as a function of applied voltage in addition to the predicted ion height
        as a function of applied voltage using the analytic model in addition to methods 1 and 2 in the paper.
    :param include_gaps: Includes or excludes spatial gap between electrodes when calculating relevant fields.
    :param figsize: Figure dimensions in inches
    :return: The trap object from the PseudopotentialPlanarTrap class.
    """
    trap = get_default_trap()
    config = get_default_config()
    parameters = ['charge_to_mass']
    bounds = [(-1.E-2, -1.E-4)]
    dc_voltages, y0, yspread, v_min, y_min, micro_min, c2m, null_volt, null_height = get_data()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    trap.v_dc = v_min
    print(f'v_dc at null: {v_min:.1f} V')
    delta_y_gradient_calc = 1.E-6
    gradient_at_null = ((trap.u_dc(trap.a / 2., y_min) - trap.u_dc(trap.a / 2, y_min - delta_y_gradient_calc)) /
                        delta_y_gradient_calc)
    gradient_at_null_low = ((trap.u_dc(trap.a / 2., y_min - trap.v_error) - trap.u_dc(trap.a / 2, y_min - delta_y_gradient_calc - trap.v_error)) /
                        delta_y_gradient_calc)
    gradient_at_null_high = ((trap.u_dc(trap.a / 2., y_min + trap.v_error) - trap.u_dc(trap.a / 2, y_min - delta_y_gradient_calc + trap.v_error)) / delta_y_gradient_calc)
    trap.charge_to_mass = g / gradient_at_null
    print("q/m from rf null: " + str(trap.charge_to_mass))
    model_voltages = np.linspace(np.min(dc_voltages), np.max(dc_voltages), num=100)
    y0_model = trap.get_height_versus_dc_voltages(model_voltages, include_gaps=include_gaps)
    method_2, = ax.plot(model_voltages, y0_model * 1.E3, color='k', label='Method 2')
    trap.charge_to_mass = g / gradient_at_null_low
    print("q/m from rf null (LOW): " + str(trap.charge_to_mass))
    trap.charge_to_mass = g / gradient_at_null_high
    print("q/m from rf null (HIGH): " + str(trap.charge_to_mass))
    guesses = [trap.__dict__[param] for param in parameters]

    def merit_func(args):
        for i, key in enumerate(parameters):
            trap.__dict__[key] = args[i]
        y0_model = trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps)
        l2 = np.sum((y0 - y0_model) ** 2)
        return l2

    res = minimize(merit_func, guesses, bounds=bounds)

    for i, param in enumerate(parameters):
        print(f'{param}: {res.x[i]}')
        trap.__dict__[param] = res.x[i]
    y0_meas = trap.get_height_versus_dc_voltages(model_voltages, include_gaps=include_gaps)
    ax.plot(dc_voltages, y0 * 1.E3, marker='.', linestyle='None', color='indigo')
    plt.errorbar(dc_voltages, y0 * 1.E3, yerr=0.0164, fmt='none', ls='none', capsize=2, color='indigo')
    method_1, = ax.plot(model_voltages, y0_meas * 1.E3, color='k', linestyle='--', label='Method 1')
    ax.set_xlabel('DC electrode voltage (V)', fontsize=12)
    ax.set_ylabel('Ion height (mm)', fontsize=12)
    ax.grid(True)
    ax.legend(handles = [method_1, method_2])
    fig.tight_layout()
    os.makedirs(config.save_path[2], exist_ok =True)
    fig.savefig(config.save_path[2]+"fig4-height_fit.pdf")
    return trap


def plot_escape(figsize=(3.5, 3)):
    """
    Plots and saves the potential energy divided by charge along the x-axis at different ion heights as a function of
    applied DC central electrode voltage.
    :param figsize: Figure dimensions in inches
    """
    trap = get_default_trap()
    config = get_default_config()
    fig, ax = plot_trap_escape_vary_dc(trap, dc_values=np.linspace(0., -300., num=11), include_gaps=True, figsize=figsize)
    ax.set_ylabel('Potential energy / charge (J/C)', fontsize=12)
    ax.set_title(None)
    plt.gca().invert_yaxis()
    fig.tight_layout()
    os.makedirs(config.save_path[2], exist_ok =True)
    fig.savefig(config.save_path[2] +"fig4-trap_escape.pdf")


def plot_height_and_micro(figsize=(3.5, 3), config = get_default_config()):
    '''
    Plots and labels the height and micromotion graphs
    '''
    voltage, height, micromotion, v_min, y_min, micro_min, c2m, minvolt_raw, RF_height = get_data()
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), height_ratios=[2, 1])

    ax1.errorbar(-voltage, micromotion*1e3, yerr=config.pixel_to_mm, color=COLORS['error'], fmt='', capsize=4, alpha=1,
                 ls='none', elinewidth=3)
    ax1.scatter(-voltage, micromotion*1e3, color=COLORS['main'], zorder=3)
    ax1.set_xlabel('Voltage (-V)')
    ax1.set_ylabel('Amplitude (mm)')
    ax1.axvline(minvolt_raw, color='black', alpha=0.6)
    ax1.annotate(f'RF null = {int(minvolt_raw)}',
                 (int(minvolt_raw), micromotion[np.abs(voltage - minvolt_raw).argmin()]), (minvolt_raw - 40, 0.25),
                 fontsize=18)

    ax2.scatter(-voltage, height*1e3, color=COLORS['main'])
    ax2.errorbar(-voltage, height*1e3, yerr=micromotion*1e3, fmt='', capsize=0, color=COLORS['main'], alpha=0.4, elinewidth=4)
    ax2.set_ylabel('Height (mm)')
    ax2.annotate(f'RF null = ({int(minvolt_raw)}, {np.round(RF_height, 2)})',
                 (minvolt_raw, RF_height), (minvolt_raw - 40, RF_height - 0.6), fontsize=18)
    ax2.axhline(RF_height, color='black', alpha=0.6)
    ax2.legend(['Height', 'RF Null', 'Micromotion'], fontsize=18, loc='upper left')
    ax2.axvline(minvolt_raw, color='black', alpha=0.6)
    if config.save_fig == True:
        os.makedirs(config.save_path[1], exist_ok =True)
        fig.savefig(str(config.save_path[1]) + 'fig3-height-micro-plot.pdf')
        print('Figure saved to "' + str(config.save_path[1]) + 'fig3-height-micro-plot.pdf"')
    plt.show()


def plot_c2m_hist(config = get_default_config()):
    '''
    Iterates over a folder to graph a histogram of charge-to-mass values
    '''
    files = os.listdir(config.hist_folder_name)
    c2m_values = []
    foldername = config.hist_folder_name
    for file_name in files:
        c2m = get_data(filename = (str(foldername) + '/' + str(file_name)))
        c2m_values.append(c2m)
    plt.figure()
    plt.hist(c2m_values, edgecolor='black', bins=22, range=(-0.003, 0), color=COLORS['main'])
    plt.axvline(x=-0.0025, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=-0.0015, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=-0.0005, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlabel('Charge-to-Mass Ratio (C/kg)')
    plt.ylabel('Number of Occurrences')
    plt.savefig(str(config.save_path[1]) + 'fig3-histogram.pdf')


if __name__ == "__main__":
    config = FigureParameterConfig()
    y_cuts_panel()
    e_field_panel()
    potential_energy_panel()
    plot_escape(figsize=(3.5, 3))
    plot_height_fit(figsize=(2.5, 3), include_gaps=True)
    plot_height_and_micro()
    plot_c2m_hist()
    plt.show()
