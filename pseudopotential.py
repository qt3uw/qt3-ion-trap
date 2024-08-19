from dataclasses import dataclass

import scipy.optimize
from scipy.constants import g
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def get_sequential_colormap(num, cmap='viridis', cmin=0.0, cmax=1.0):
    xs = np.linspace(cmin, cmax, num=num)
    cs = mpl.colormaps[cmap]
    return [cs(x) for x in xs]

@dataclass
class PseudopotentialPlanarTrap:
    central_electrode_width: float = 3.175E-3
    central_electrode_gap: float = 1.E-3
    ac_electrode_width: float = 4.15831E-3
    ac_electrode_inner_gap: float = 1.E-3
    v_rf: float = 0.5 * 50 * 75
    v_dc: float = 200.
    charge_to_mass: float = 6.8E-4
    freq_rf: float = 60.
    gap_width: float = 2.E-3

    @property
    def a(self):
        return self.central_electrode_width + 2 * self.central_electrode_gap

    @property
    def b(self):
        return self.ac_electrode_width + 2 * self.ac_electrode_inner_gap

    @property
    def c(self):
        return self.b

    @property
    def x1(self):
        return 0

    @property
    def x2(self):
        return self.a

    @property
    def omega(self):
        return 2 * np.pi * self.freq_rf

    @property
    def height_no_gap(self):
        return np.sqrt(2 * self.a * self.b + self.a ** 2) / 2

    @property
    def height_with_gap(self):
        return np.sqrt(2 * self.a * self.b + self.a ** 2 - self.gap_width ** 2) / 2

    @property
    def y_escape(self):
        raise NotImplementedError


    def phi_ac(self, x, y):
        """
        Calculates the free-space potential from the AC electrode.
        :param x:
        :param y:
        :return:
        """
        return (self.v_rf / np.pi)*(np.arctan((-self.c-x)/y) - np.arctan((-self.gap_width - x)/y)
                + np.arctan(((self.a+self.b-self.gap_width / 2)-x)/y) - np.arctan(((self.a + self.gap_width / 2) - x)/y))

    def phi_diel_i(self, x1, x2, x, y, delta_V):
        """
        Calculates the free-space potential from the dielectric between electrodes.
        :param x:
        :param y:
        :return:
        """
        return ((delta_V) / (np.pi * (x1 - x2))) * ((y/2)*np.log(((x-x1)**2 + y**2) /
                (x-x2)**2 + y**2) + (x-x1) * (np.arctan((x-x2/y)) - np.arctan((x - x1)/y)))

    def grad_phi(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        grad_x = (self.v_rf * (1. / (y * ((x + self.c) ** 2 / y ** 2 + 1)) - 1 / (y * (x ** 2 / y ** 2 + 1)) - 1 / (
                    y * ((-x + self.b + self.a) ** 2 / y ** 2 + 1)) + 1 / (y * ((self.a - x) ** 2 / y ** 2 + 1)))) / np.pi
        grad_y = (self.v_rf * (-(x + self.c) / (((x + self.c) ** 2 / y ** 2 + 1) * y ** 2) + x / ((x ** 2 / y ** 2 + 1) * y ** 2) - (
                    -x + self.b + self.a) / (((-x + self.b + self.a) ** 2 / y ** 2 + 1) * y ** 2) + (self.a - x) / (
                                     ((self.a - x) ** 2 / y ** 2 + 1) * y ** 2))) / np.pi
        return grad_x, grad_y

    def u_ac(self, x, y):
        """
        Returns pseudopotential from the AC electrodes, divided by the charge to mass ratio
        :param x:
        :param y:
        :return:
        """
        gradx, grady = self.grad_phi(x, y)
        return (1. / (4 * self.omega ** 2)) * (gradx ** 2 + grady ** 2) * self.charge_to_mass

    def u_dc(self, x, y):
        """
        Returns potential from the DC electrodes, divided by the charge to mass ratio

        :param x:
        :param y:
        :return:
        """
        return (self.v_dc / np.pi) * (np.arctan((self.a - self.gap_width/2 - x) / y) - np.arctan(((0)- x) / y))
    def u_gap(self, x, y):

        return  [phi_diel_i((-self.c - self.gap_width), -self.c, x, y, self.v_rf), phi_diel_i(-g, 0, x, y, self.v_dc - self.v_rf),
                 phi_diel_i(self.a - self.gap_width / 2, self.a + self.gap_width / 2, x, y, self.v_rf - self.v_dc),
                phi_diel_i(self.a + self.b - self.gap_width / 2, self.a + self.b + self.gap_width / 2, x, y, - self.v_rf)]
    def u_gravity(self, x, y):
        """
        Returns gravitational potential, divided by the charge to mass ratio

        :param x:
        :param y:
        :return:
        """
        return (1. / self.charge_to_mass) * g * y

    def u_total(self, x, y):
        """
        Returns sum of gravitational, dc, and ac pseudopotential divided by charge
        :param x:
        :param y:
        :return:
        """
        return self.u_gravity(x, y) + self.u_dc(x, y) + self.u_ac(x, y) + \
                (self.u_gap(x, y)[0] + self.u_gap(x, y)[1] + self.u_gap(x, y)[2] + self.u_gap(x, y)[3])

    def plot_potential_contours(self, x_range=[-7.5E-3, 12.5E-3], y_range=[1.E-3, 6.E-3], resolution=[512, 512],
                                fig=None, ax=None, ncountours=40, max_countour_level=300.):
        x = np.linspace(x_range[0], x_range[1], num=resolution[0])
        y = np.linspace(y_range[0], y_range[1], num=resolution[1])
        x, y = np.meshgrid(x, y)
        u = self.u_total(x, y)
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        levels = np.linspace(0., max_countour_level, num=ncountours)
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
            cs = ax.imshow(np.flipud(u), extent=[ex * 1.E3 for ex in extent])
            ax.contour(u, levels=levels, colors='k', extent=[ex * 1.E3 for ex in extent])
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('height (mm)')
            ax.set_title(f'potential energy / charge (V)\n center electrode at {trap.v_dc:.1f} V')
            fig.colorbar(cs)

        return fig, ax

    def plot_y_cuts(self, yrange=(1.E-3, 10.E-3), num=200, x0=None):
        y = np.linspace(*yrange, num=num)
        if x0 is None:
            x0 = trap.a / 2
        x = np.zeros_like(y) + x0


        v_grav = trap.u_gravity(x, y)
        phi_dc = trap.u_dc(x, y)
        psuedo = trap.u_ac(x, y)

        fig, ax = plt.subplots(1, 1)
        ax.plot(y * 1.E3, psuedo, label='psuedo')
        ax.plot(y * 1.E3, phi_dc, label='dc')
        ax.plot(y * 1.E3, v_grav, label='gravity')
        ax.plot(y * 1.E3, v_grav + phi_dc + psuedo, label='total')
        fig.legend()
        ax.set_xlabel('y (mm)')
        ax.grid()
        ax.set_ylabel('potential energy / charge (V)')

    def find_equilibrium_height(self):
        def merit_func(y):
            return self.u_total(self.a / 2., y)
        res = minimize_scalar(merit_func, bounds=(1.E-4, 1.))
        return res.x

    def get_height_versus_dc_voltages(self, dc_voltages):
        dc_initial = self.v_dc
        y0 = []
        for v_dc in dc_voltages:
            self.v_dc = v_dc
            y0.append(self.find_equilibrium_height())

        self.v_dc = dc_initial
        return np.array(y0)


def plot_trap_escape_vary_dc(trap: PseudopotentialPlanarTrap, dc_values=np.linspace(0., 150., num=20), xrange=[-7.4E-3, 12.4E-3], xnum=100):
    fig, ax = plt.subplots(1, 1)
    xs = np.linspace(xrange[0], xrange[1], num=xnum)

    y0s = []
    colors = get_sequential_colormap(len(dc_values))
    for i, v_dc in enumerate(dc_values):
        trap.v_dc = v_dc
        y0s.append(trap.find_equilibrium_height())
        ax.plot(xs * 1.E3, trap.u_total(xs, y0s[-1]), label=f'{v_dc:.1f} V', color=colors[i])
    ax.grid()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('potential energy / charge at equilibrium height (V)')
    fig.legend()

    figh, axh = plt.subplots(1, 1)
    axh.plot(dc_values, y0s, marker='o', linestyle='None')
    axh.set_xlabel('DC electrode voltage (V)')
    axh.set_ylabel('equilibrium height (mm)')
    axh.grid()

def get_data():
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
    return dc_voltages, y0 * 1.E-3, y_spread * 1.E-3


def fit_data(trap: PseudopotentialPlanarTrap, parameters, bounds=None):
    dc_voltages, y0, yspread = get_data()
    fig, ax = plt.subplots(1, 1)
    guesses = [trap.__dict__[param] for param in parameters]

    def merit_func(args):
        for i, key in enumerate(parameters):
            trap.__dict__[key] = args[i]
        y0_model = trap.get_height_versus_dc_voltages(dc_voltages)
        return np.sum((y0 - y0_model) ** 2)

    res = minimize(merit_func, guesses, bounds=bounds)

    for i, param in enumerate(parameters):
        print(f'{param}: {res.x[i]}')
        trap.__dict__[param] = res.x[i]

    model_voltages = np.linspace(np.min(dc_voltages), np.max(dc_voltages), num=100)
    y0_model = trap.get_height_versus_dc_voltages(model_voltages)

    ax.plot(dc_voltages, y0 * 1.E3, marker='o', linestyle='None')
    ax.plot(model_voltages, y0_model * 1.E3)
    ax.set_xlabel('DC electrode voltage (V)')
    ax.set_ylabel('ion height (mm)')
    ax.grid()

    return trap


if __name__ == "__main__":
    trap = PseudopotentialPlanarTrap()
    trap.charge_to_mass = 0.9E-3
    fit_data(trap, ['charge_to_mass', 'central_electrode_gap'], bounds=[(1.E-4, 1.E-2), (.1E-3, 4E-3)])
    # trap.v_dc = 90.
    #
    trap.plot_potential_contours(y_range=(0.5E-3, 10.E-3))
    print(trap.central_electrode_gap)
    # # trap.plot_y_cuts()
    plot_trap_escape_vary_dc(trap, dc_values=np.linspace(0., 230., num=20))
    # print(trap.find_equilibrium_height())


    plt.show()
    #
    # figim, axim = plt.subplots(1, 1)
