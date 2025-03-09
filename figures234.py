import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chisquare
from scipy.constants import g
import os
import math as math
from matplotlib import colormaps

from matplotlib.backends.backend_pdf import PdfPages

from pseudopotential import PseudopotentialPlanarTrap, plot_trap_escape_vary_dc, get_sequential_colormap
from acquisition.uncertainties import Uncertainties
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

TRIAL = str(11)

class FigureParameterConfig:
    def __init__(self, uncert, height_file_name = "data/raw_micromotion/second_round_data_collection/Clean Data/02-28-2025_Trial" + TRIAL + "_data.txt", save_path = ["figures/figure_" + str(i) + "/" for i in range(2, 5)]):
        self.save_fig = True                                                    # Saves figure to directory specified by self.save_path
        self.unc = uncert
        self.pixel_to_mm = uncert.pxl_to_mm                                      # Pixel to mm conversion from calibration. Only for plotting error bars, okay to set to zero if trials vary
        # pixel_to_mm = 0.0164935065 
        self.graph_file_name = height_file_name     # File to plot height & micromotion vs. voltage graphs for
        self.hist_folder_name = "data/analyzed_micromotion/second_round_data_collection/"                     # Folder to extract charge-to-mass values from and graph the histogram
        self.save_path = save_path                                                # Path for exported figures


def get_default_config(height_file_name = "data/raw_micromotion/second_round_data_collection/Clean Data/02-28-2025_Trial" + str(11) + "_data.txt", save_path = ["figures/figure_" + str(i) + "/" for i in range(2, 5)], uncert = Uncertainties(r_c = [np.nan, 16.053-0.178], N_c = [np.nan, 900-10])):
    return FigureParameterConfig(uncert = uncert, height_file_name = height_file_name, save_path = save_path)

  
def get_default_trap():
    """
    Creates a and returns a trap object
    :return: A trap object from the PseudopotentialPlanarTrap class
    """
    trap = PseudopotentialPlanarTrap()
    trap.v_rf = 47 * -20 * np.sqrt(2)
    trap.charge_to_mass = -1.077E-3
    return trap

def y_cuts_panel():
    """
    Plots and saves the potential energy divided by charge of the various relevant scalar fields
    """
    config = get_default_config()
    trap = get_default_trap()
    trap.v_dc = -80.
    fig, ax = trap.plot_y_cuts(include_gaps=True, figsize=(12, 7), mult_range = range(1, 20))
    fig.tight_layout()
    os.makedirs(config.save_path[0], exist_ok =True)
    """
    for i in range(1, 40):
        if i != 0:
            fig.savefig(config.save_path[0] +"fig2-y-cuts" + str(i) + ".pdf")
            trap.charge_to_mass = i * (-1.077E-3)
            trap.plot_y_cuts(include_gaps=True, figsize=(3.5, 3))
    """

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

    
def get_data(config, filename = None):
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
        analyzedfilename = 'data/analyzed_micromotion/second_round_data_collection/' + str(cut_basefilename) + '_analyzed.txt'
        with open(datafile, 'r') as file:
            for line in file:
                line = line.strip().replace('[', '').replace(']', '')
                data_list.append([float(value) for value in line.split(',')])
            rawdata = np.array(data_list)[ : , [0, 1, 3]]
            std_data = np.array(data_list)[ : , [2, 4]]
            print("Raw data: " + str(rawdata))
            dc_voltages = rawdata[:, 0]

            y_spread = rawdata[:, 2]
            y0 = rawdata[:, 1]
            print("y0: " + str(y0))
            v_min, y_min, micro_min = rawdata[np.argmin(rawdata[:, 2])]
            y_std = std_data[:, 0]
            spread_std = std_data[:, 1]
            print("y_std: " + str(y_std))
            print("micro_std: " + str(spread_std))
        with open(analyzedfilename) as file:
            for line in file:
                line = line.strip()
                analyzed_data = eval(line)
                c2m, null_volt, null_height = analyzed_data[0], analyzed_data[1], analyzed_data[2]
        return -dc_voltages, y0 * 1.E-3, y_std *  1.E-3, y_spread * 1.E-3, spread_std * 1.E-3, -v_min, y_min * 1.E-3, micro_min * 1.E-3, c2m, null_volt, null_height
    else:
        analyzedfilename = 'data/analyzed_micromotion/second_round_data_collection/' + str(cut_basefilename) + '.txt'
    with open(analyzedfilename) as file:
        for line in file:
            line = line.strip()
            analyzed_data = eval(line)
            c2m, null_volt, null_height = analyzed_data[0], analyzed_data[1], analyzed_data[2]
    return c2m
    

def plot_height_fit(config, include_gaps=True, figsize=(3.5, 3)):
    """
    Plots and saves experimental ion height as a function of applied voltage in addition to the predicted ion height
        as a function of applied voltage using the analytic model in addition to methods 1 and 2 in the paper.
    :param include_gaps: Includes or excludes spatial gap between electrodes when calculating relevant fields.
    :param figsize: Figure dimensions in inches
    :return: The trap object from the PseudopotentialPlanarTrap class.
    """
    trap = get_default_trap()
    parameters = ['charge_to_mass']
    bounds = [(-1.E-2, -1.E-5)]
    dc_voltages, y0, y_std, yspread, spread_std, v_min, y_min, micro_min, c2m, null_volt, null_height = get_data(config = config)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    trap.v_dc = -null_volt
    print(dc_voltages)
    print(y0)
    print(f'v_dc at null: {v_min:.1f} V')
    delta_y_gradient_calc = 1.E-6

    # Calculate the gradient with clearer sign convention
    gradient_at_null = -trap.grad_u_dc(x = trap.a / 2., y = null_height* 10**-3, x1 = trap.x1())
    lapl_at_null =  (trap.grad_u_dc(x = trap.a / 2., y = null_height* 10**-3 + delta_y_gradient_calc, x1 = trap.x1()) - (trap.grad_u_dc(x = trap.a / 2., y = null_height* 10**-3, x1 = trap.x1())))/delta_y_gradient_calc
    #gradient_at_null = ((trap.u_dc(trap.a / 2., null_height * 10**-3 + delta_y_gradient_calc) - trap.u_dc(trap.a / 2, null_height * 10 ** -3)) / delta_y_gradient_calc)
    
    # Print debugging information
    print(f'Gradient at RF null: {gradient_at_null:.3e} V/m')
    print(f'v_dc at null: {v_min:.1f} V')
    
    # Since gravity pulls down, a positive gradient (pushing up for negative charge) balances gravity
    # For a negative charge-to-mass ratio, the gradient should be positive for stable equilibrium
    # The formula should be: q/m = g / grad_E (for positive grad_E and negative q/m)
    c2m_ext = -g / abs(gradient_at_null) 
    r_dev = [[np.nan], y_std]
    uncertain = config.unc
    print(uncertain.delta_pos_calc(r_sta = r_dev))
    c2m_err = np.sqrt((np.abs(c2m_ext**2 / g *   1/np.abs(lapl_at_null)) * r_dev[1])) 
    print(c2m_err)
    trap.charge_to_mass = -g / abs(gradient_at_null) 

    
    print(f"q/m from RF null: {trap.charge_to_mass:.3e}")
    print("q/m from rf null: " + str(trap.charge_to_mass))

    v_ans = (-trap.u_total(trap.a / 2, y_min) / trap.u_dc(trap.a/2, y_min) + 1)
    dc_voltages_fine = np.linspace(start = dc_voltages[-1], stop = dc_voltages[0], num = 100)
    print(dc_voltages_fine)
    print("c2m: " + str(c2m_ext))
    y0_model = (trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps)) 
    print("c2m Upper: " + str(c2m_ext -c2m_err))
    trap.charge_to_mass = c2m_ext -c2m_err
    y0_model_upper = (trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps)) 
    trap.charge_to_mass = c2m_ext +c2m_err
    print("c2m Lower: " + str(c2m_ext +c2m_err))
    y0_model_lower= (trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps)) 
    
    #print("q/m from rf null (LOW): " + str(trap.charge_to_mass))

    #print("q/m from rf null (HIGH): " + str(trap.charge_to_mass))
    guesses = [trap.__dict__[param] for param in parameters]

    def merit_func(args):
        for i, key in enumerate(parameters):
            trap.__dict__[key] = args[i]
        y0_model = trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps)
        l2 = np.sum((y0 - y0_model) ** 2)
        return l2

    def chi_2(y_exp, y_fit, sigma):
        return np.sum(np.divide(np.square(y_exp - y_fit), np.square(sigma)))

    def fit_error(y_fit, sigma, trap = trap):
        V_inv = np.asmatrix(np.diag(np.square(sigma))).I
        F1 = trap.u_gravity(np.ones_like(y_fit) * trap.a / 2, y_fit)
        F2 = trap.u_ac(np.ones_like(y_fit) * trap.a / 2 * trap.a / 2, y_fit)
        #F3 = [trap.u_dc(trap.a / 2, y_i) for y_i in y_fit]
        F = np.asmatrix([F1, F2])
        U =  (F @ V_inv @ F.T).I
        return U

    res = minimize(merit_func, guesses, bounds=bounds)
    for i, param in enumerate(parameters):
        print(f'{param}: {res.x[i]}')
        trap.__dict__[param] = res.x[i]
    trap.charge_to_mass = res.x
    print("c2m_fit: " + str(trap.charge_to_mass))
    y0_meas = trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps) 
    delta_y = 0.0164
    sys_err = np.ones_like(y_std) * delta_y + y_std
    y_std_chi = np.sqrt(np.square(28/64 * 0.005 * 25.4* y0) + np.square(y_std))/3
    print(y0)
    print(y_std_chi)
    chi2_fit = chi_2(y0, y0_meas, y_std_chi)
    chi2_extr = chi_2(y0, y0_model, y_std_chi)
    print('chi_2 value (fit): ' + str( chi2_fit))
    print('chi_2 value (extrapolation): ' + str(chi2_extr))

    error = fit_error(y0_meas,y_std_chi, trap=trap)
    trap_c2m_0 = trap.charge_to_mass 
    c2m_int = trap_c2m_0
    chi2_fit_upper =  trap_c2m_0 + 3*np.sqrt(error[1, 1])
    print(np.sqrt(error[1, 1]))
    print(chi2_fit_upper)


    chi2_fit_lower =  trap_c2m_0 - 3*np.sqrt(error[1, 1])
    trap.charge_to_mass = chi2_fit_upper

    print(chi2_fit_lower)

    y0_meas_upper = trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps) 
    trap.charge_to_mass = chi2_fit_lower
    y0_meas_lower = trap.get_height_versus_dc_voltages(dc_voltages, include_gaps=include_gaps) 
    print(error)
    print(y0_meas_upper)
    ax.plot(dc_voltages, (y0)* 1.E3, marker='.', linestyle='None', color='k')
    ax.plot(-null_volt, null_height, marker = '.', color = "red")
    ax.plot(dc_voltages,(y0_meas * 1.E3), color='darkred', linestyle='--', label='Method 1: ' + r'$\chi^{2}_{1} = $ ' + "{:.3f}".format(chi2_fit))
    ax.fill_between(dc_voltages,y0_meas_upper * 1.E3, y0_meas_lower*1.E3, alpha = .3, hatch = '///', color = 'red')
    #ax.plot(dc_voltages,(y0_meas_lower * 1.E3), color='yellow', linestyle='--', label='Method 1 Upper ')
    #ax.plot(dc_voltages,(y0_meas_upper * 1.E3), color='blue', linestyle='--', label='Method 1 Lower: ')
    plt.errorbar(dc_voltages,(y0)* 1.E3, yerr=(np.sqrt(((28/64 * 0.005) * 25.4 * 1.E3 * y0)**2 + (y_std * 3 * 1.E3)**2)), fmt='none', ls='none', capsize=2, color='indigo')
    
    ax.plot(dc_voltages,(y0_model * 1.E3), color= "green", label='Method 2: ' + r'$\chi^{2}_{2} = $ ' + "{:.3f}".format(chi2_extr))
    #ax.plot(dc_voltages,(y0_model_upper * 1.E3), color='teal', label='Method 2: ' + r'$\chi^{2}_{2} = $ ' + "{:.3f}".format(chi2_extr))
    #ax.plot(dc_voltages,(y0_model_lower * 1.E3), color='green', label='Method 2: ' + r'$\chi^{2}_{2} = $ ' + "{:.3f}".format(chi2_extr))
    ax.fill_between(dc_voltages,y0_model_upper * 1.E3, y0_model_lower*1.E3, alpha = .2, color = COLORS["error"])
    ax.set_xlabel('DC electrode voltage (V)', fontsize=12)
    ax.set_ylabel('Ion height (mm)', fontsize=12)
   
    ax.grid(True)
    #ax.legend(handles = [method_1, method_1u, method_1l, method_2])
    fig.tight_layout()
    os.makedirs(config.save_path[2], exist_ok =True)
    metadata = {"data Source": config.graph_file_name, "charge-to-mass_interpolated" : str(c2m_int[0]), "c2m_int_err" : str(3 / np.sqrt(error[1, 1])), "charge-to-mass_extrapolated" : str(c2m_ext), "c2m_ext_err" : str(c2m_err), \
                "chi^2 _fit" : str(chi2_fit), "chi^2_ext)" : str(chi2_extr)}
    fig.savefig(config.save_path[2]+"fig4-height_fit_Trial" + TRIAL + ".pdf", metadata = metadata)


    return trap



def plot_height_and_micro(config, figsize=(3.5, 3)):
    '''
    Plots and labels the height and micromotion graphs
    '''
    voltage, height, height_std, micromotion, micro_std, v_min, y_min, micro_min, c2m, minvolt_raw, RF_height = get_data(config = config)
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), height_ratios=[2, 1])

    ax1.errorbar(-voltage, micromotion*1e3, yerr=((28/64 * 0.005) * 25.4 * micromotion * 1e3 +  micro_std * 1e3 * 3), color=COLORS['error'], fmt='', capsize=4, alpha=1,
                 ls='none', elinewidth=3)
    ax1.scatter(-voltage, micromotion*1e3, color=COLORS['main'], zorder=3)
    ax1.set_xlabel('Voltage (-V)')
    ax1.set_ylabel('Amplitude (mm)')
    ax1.axvline(minvolt_raw, color='black', alpha=0.6)
    ax1.annotate(f'RF null = {int(minvolt_raw)}',
                 (int(minvolt_raw), micromotion[np.abs(voltage - minvolt_raw).argmin()]), (minvolt_raw, 0.25),
                 fontsize=18)

    ax2.scatter(-voltage, height*1e3, color=COLORS['main'])
    ax2.errorbar(-voltage, height*1e3, yerr=micromotion*1e3, fmt='', capsize=0, color=COLORS['main'], alpha=0.4, elinewidth=4)
    ax2.set_ylabel('Height (mm)')
    #ax2.annotate(f'RF null = ({int(minvolt_raw)}, {np.round(RF_height, 2)})',
                 #(minvolt_raw, RF_height), (minvolt_raw, RF_height), fontsize=18)
    ax2.axhline(RF_height, color='black', alpha=0.6)
    ax2.legend(['Height', 'RF Null', 'Micromotion'], fontsize=18, loc='upper left')
    ax2.axvline(minvolt_raw, color='black', alpha=0.6)
    if config.save_fig == True:
        os.makedirs(config.save_path[1], exist_ok =True)
        fig.savefig(str(config.save_path[1]) + 'fig3-height-micro-plot.pdf')
        print('Figure saved to "' + str(config.save_path[1]) + 'fig3-height-micro-plot.pdf"')
    plt.show()


def plot_c2m_hist(config):
    '''
    Iterates over a folder to graph a histogram of charge-to-mass values
    '''
    files = os.listdir(config.hist_folder_name)
    c2m_values = []
    foldername = config.hist_folder_name
    for file_name in files:
        c2m = get_data(config = config, filename = (str(foldername) + '/' + str(file_name)))
        c2m_values.append(-c2m)

    plt.figure()
    plt.hist(c2m_values, edgecolor='black', bins=18, range=(-0.003, 0), color=COLORS['main'])
    plt.axvline(x=-0.0025, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=-0.0015, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=-0.0005, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlabel('Charge-to-Mass Ratio (C/kg)')
    plt.ylabel('Number of Occurrences')
    plt.savefig(str(config.save_path[1]) + 'fig3-histogram.pdf')


if __name__ == "__main__":
 
    
    # y_cuts_panel()
    # e_field_panel()
    # potential_energy_panel()
    # plot_escape(figsize=(3.5, 3))
    U = Uncertainties(r_c = [np.nan, 16.053-0.178], N_c = [np.nan, 900-10], delta_r_c = [np.nan, 1/(2*np.sqrt(12))])
    for i in [2, 5, 6, 7,  11, 12, 13, 14, 16, 17, 19]:
        configure = get_default_config(height_file_name = "data/raw_micromotion/second_round_data_collection/Clean Data/02-28-2025_Trial" + str(i) + "_data.txt", \
           save_path = ["figures/figure_" + str(j) + "/02-28-2025/Trial" + str(i) + "/numeric_grad_u_dc/"  for j in range(2, 5)], uncert = U) 
        plot_height_fit(config = configure)
        plot_height_and_micro(config = configure)
    plot_c2m_hist(config = get_default_config())
    plt.show()



    