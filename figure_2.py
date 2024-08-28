import matplotlib.pyplot as plt
import numpy as np

from pseudopotential import PseudopotentialPlanarTrap

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
    trap.charge_to_mass = 2.1E-3
    return trap

def y_cuts_panel():
    trap = PseudopotentialPlanarTrap()
    trap.charge_to_mass = 2.1E-3
    fig, ax = trap.plot_y_cuts(include_gaps=False, figsize=(3.5, 3))
    fig.tight_layout()
    fig.savefig('y-cuts.pdf')

def e_field_panel():
    trap = get_default_trap()
    trap.v_ac = 1000.
    figp, axp = trap.plot_E_field(include_gaps=True, figsize=(3.5, 3))
    figp.tight_layout()
    figp.savefig('efield_positive.pdf')
    trap.v_ac = -1000.
    fign, axn = trap.plot_E_field(include_gaps=True, figsize=(3.5, 3))
    fign.tight_layout()
    fign.savefig('efield_negative.pdf')

if __name__ == "__main__":
    y_cuts_panel()
    e_field_panel()
    # plt.show()