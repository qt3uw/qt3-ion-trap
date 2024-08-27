from dataclasses import dataclass

import scipy.optimize
from scipy.constants import g
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import linalg as LA

def get_sequential_colormap(num, cmap='viridis', cmin=0.0, cmax=1.0):
    xs = np.linspace(cmin, cmax, num=num)
    cs = mpl.colormaps[cmap]
    return [cs(x) for x in xs]

@dataclass
class PseudopotentialPlanarTrap:
    central_electrode_width: float = 3.175E-3
    ac_electrode_width: float = 4.15831E-3
    v_rf: float = 0.5 * 50 * 75
    v_dc: float = 200.
    charge_to_mass: float = 6.8E-4
    freq_rf: float = 60.
    gap_width: float = 2.E-3

    @property
    def a(self):
        return self.central_electrode_width + self.gap_width

    @property
    def b(self):
        return self.ac_electrode_width + self.gap_width

    @property
    def c(self):
        return self.b

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

    @staticmethod
    def phi_diel_i(x, y, x_1, x_2, dv):
        """
        :param x: x-coordinate
        :param y: y-coordinate
        :param x_1: gap boundary
        :param x_2: gap boundary
        :param dv: voltage across gap
        :return:
        """
        return dv / (np.pi * (x_1 - x_2)) * (
                    (y / 2) * np.log(((x - x_1) ** 2 + y ** 2) / ((x - x_2) ** 2 + y ** 2)) + (x - x_1) * (
                        np.arctan((x - x_2) / y) - np.arctan((x - x_1) / y)))

    @staticmethod
    def phi_electrode(x, y, x1, x2, v):
        """
        Calculates the free-space potential from a rectangular electrode infinite in z, where y is normal to electrode
        :param x: x-coordinates
        :param y: y-coordinates
        :param x1: electrode boundary
        :param x2: electrode boundary
        :param v: electrode voltage
        :return:
        """
        return v / np.pi * (np.arctan((x2 - x) / y) - np.arctan((x1 - x) / y))

    def phi_gaps_linear(self, x, y):
        """
        Gets the potential from each of the gaps, modelling as a linear interpolation between neighboring electrodes
        :param x: x-coordinates
        :param y: y-coordinates
        :return:
        """
        return self.phi_diel_i(x, y, -self.c - self.gap_width / 2, -self.c + self.gap_width / 2, self.v_rf) + \
                self.phi_diel_i(x, y, self.gap_width / 2,-self.gap_width/2, self.v_dc - self.v_rf) + \
            self.phi_diel_i(x, y, self.a - (self.gap_width / 2), self.a + (self.gap_width / 2), self.v_rf - self.v_dc) + \
                self.phi_diel_i(x, y, self.a + self.b + self.gap_width/2, self.a + self.b - self.gap_width/2, -self.v_rf)

    def x1(self, include_gaps=True):
        if include_gaps:
            return self.gap_width / 2
        else:
            return 0.

    def x2(self, include_gaps=True):
        if include_gaps:
            return self.central_electrode_width + self.gap_width / 2
        else:
            return self.a
    def phi_ac(self, x, y):
        """
        This potential includes the AC electrode and neighboring gaps
        :param x:
        :param y:
        :return:
        """
        return self.phi_electrode(x, y, -self.c, 0, self.v_rf) + self.phi_electrode(x, y, self.a, self.a + self.b, self.v_rf)

    def phi_ac_with_gaps(self, x, y):
        """
        AC potential including the interpolated gap voltages
        :param x:
        :param y:
        :return:
        """
        return self.phi_electrode(x, y, -self.c + 0.5 * self.gap_width, -0.5 * self.gap_width, self.v_rf) + \
            self.phi_electrode(x, y, self.a + 0.5 * self.gap_width,
                               self.a + self.ac_electrode_width + 0.5 * self.gap_width, self.v_rf) + \
            self.phi_gaps_linear(x, y)

    def grad_phi_ac_gaps(self, x, y):
        """
        Numerically computes the gradient including a linearly varying voltage across the insulating gaps
        :param x: x-coordinates (can be 1D or 2D)
        :param y: y-coordinates (can be 1D or 2D)
        :return:
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if x.ndim == 1:
            x = np.reshape(x, (1, -1))  # Reshape to 2D with one row if 1D
        if y.ndim == 1:
            y = np.reshape(y, (1, -1))  # Reshape to 2D with one column if 1D
        dx =np.diff(x, axis=1).mean()
        dy = np.diff(y, axis=0).mean() if y.shape[0] > 1 else np.diff(y, axis=1).mean()  # Handle single row case
        gradx = np.zeros_like(x.flatten())
        grady = np.zeros_like(x.flatten())
        phi_ac = self.phi_ac_with_gaps(x, y)
        if x.shape[0] == 1:
            phi_ac = np.append(phi_ac, phi_ac, axis=0)
            if dx == 0:
                gradx = np.zeros_like(x.flatten())
                grady = np.gradient(phi_ac, dy)[1][0, :]
            elif dy == 0:
                gradx = np.gradient(phi_ac, dx)[1][0, :]
                grady = np.zeros_like(x.flatten())

        else:
            grady, gradx = np.gradient(phi_ac, dy, dx)  # Calculate gradients

        return gradx, grady


    def grad_phi_ac(self, x, y):
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

    def u_ac(self, x, y, include_gaps=True):
        """
        Returns pseudopotential from the AC electrodes, divided by the charge to mass ratio
        :param x:
        :param y:
        :return:
        """
        if include_gaps:
            gradx, grady = self.grad_phi_ac_gaps(x, y)
        else:
            gradx, grady = self.grad_phi_ac(x, y)

        # These three lines deal with the possibility of a 1-D array as input when including gaps, dx or dy will be nan.
        mx = np.isnan(gradx)
        my = np.isnan(grady)
        grad_squared = np.where(mx & my, np.nan, np.where(mx, 0, gradx) ** 2 + np.where(my, 0, grady) ** 2)

        return (1. / (4. * self.omega ** 2)) * grad_squared * self.charge_to_mass


    def u_dc(self, x, y, include_gaps = True):
        """
        Returns potential from the DC electrodes, divided by the charge to mass ratio

        :param x:
        :param y:
        :return:
        """
        if include_gaps:
            return (self.v_dc / np.pi) * (np.arctan((self.a - self.gap_width/2 - x) / y) - np.arctan(((self.gap_width/2)- x) / y))
        else:
            return (self.v_dc / np.pi) * (np.arctan((self.a - x) / y) - np.arctan(((- x) / y)))
    def grad_u_dc(self, x, y, include_gaps = True):
        return ((self.v_dc * (-((self.x2(include_gaps=include_gaps) - x) * 1) / ((((self.x2(include_gaps=include_gaps) - x) * 1) ** 2 / (y * 1) ** 2 + 1) * (y * 1) ** 2) - (x * 1) / ((((x * 1) / (y * 1) - (self.x1(include_gaps) * 1)) ** 2 +1) * (y * 1) ** 2))) /np.pi)
    def u_gravity(self, x, y):
        """
        Returns gravitational potential, divided by the charge to mass ratio

        :param x:
        :param y:
        :return:
        """
        return (1. / self.charge_to_mass) * g * y

    def u_total(self, x, y, include_gaps=True):
        """
        Returns sum of gravitational, dc, and ac pseudopotential divided by charge
        :param x:
        :param y:
        :return:
        """
        return self.u_gravity(x, y) + self.u_dc(x, y, include_gaps=include_gaps) + self.u_ac(x, y, include_gaps=include_gaps)

    def plot_potential_at_surface(self, num=256):
        """
        :param num: Number of samples
        :return: (fig, ax)
        """
        xlin = np.linspace(-2 * trap.c, trap.c + trap.a + trap.b, num=256)
        ys = np.zeros_like(xlin)
        figv, axv = plt.subplots(1, 1)
        axv.plot(xlin, trap.phi_ac(xlin, ys) + trap.u_dc(xlin, ys, include_gaps=False), label='ignore gaps')
        axv.plot(xlin, trap.phi_ac_with_gaps(xlin, ys) + trap.u_dc(xlin, ys, include_gaps=True), label='include gaps')
        axv.grid()
        figv.legend()
        return figv, axv


    def plot_potential_contours(self, x_range=(-10E-3, 15E-3), y_range=(1.E-3, 6.E-3), resolution=(512, 512),
                                fig=None, ax=None, ncountours=25, max_countour_level=250., include_gaps=True, **kwargs):
        x = np.linspace(x_range[0], x_range[1], num=resolution[0])
        y = np.linspace(y_range[0], y_range[1], num=resolution[1])
        x, y = np.meshgrid(x, y)
        # u = self.u_gap(x, y)[0] + self.u_gap(x, y)[1] + self.u_gap(x, y)[2] + self.u_gap(x, y)[3]
        # u = self.u_dc(x, y) + self.u_ac(x, y)
        u = self.u_total(x, y, include_gaps=include_gaps)

        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        levels = np.linspace(0., max_countour_level, num=ncountours)
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
            cs = ax.imshow(np.flipud(u), extent=[ex * 1.E3 for ex in extent], **kwargs)
            ax.contour(u, levels=levels, colors='k', extent=[ex * 1.E3 for ex in extent])
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('height (mm)')
            ax.set_title(f'potential energy / charge (V)\n center electrode at {self.v_dc:.1f} V')
            fig.colorbar(cs)

        return fig, ax

    def plot_y_cuts(self, yrange=(1.E-3, 10.E-3), num=200, x0=None, include_gaps=True):
        """


        :param yrange:
        :param num:
        :param x0:
        :param include_gaps:
        :return:
        """
        y = np.linspace(*yrange, num=num)
        if x0 is None:
            x0 = self.a / 2
        x = np.zeros_like(y) + x0


        u_grav = self.u_gravity(x, y)
        u_dc = self.u_dc(x, y)
        u_ac = self.u_ac(x, y, include_gaps=include_gaps)

        fig, ax = plt.subplots(1, 1)
        ax.plot(y * 1.E3, u_ac, label='psuedo')
        ax.plot(y * 1.E3, u_dc, label='dc')
        ax.plot(y * 1.E3, u_grav, label='gravity')
        ax.plot(y * 1.E3, u_ac + u_dc + u_grav, label='total')
        fig.legend()
        ax.set_xlabel('y (mm)')
        ax.grid()
        ax.set_ylabel('potential energy / charge (V)')
        return fig, ax

    def find_equilibrium_height(self, ystep=1.E-5, include_gaps=True):
        def merit_func(y):
            ys = np.linspace(y-ystep, y+ystep, num=3)
            xs = np.zeros_like(ys) + self.a / 2.
            return self.u_total(xs, ys, include_gaps=include_gaps).flatten()[1]
        res = minimize_scalar(merit_func, bounds=(1.E-4, 1.))
        return res.x

    def get_height_versus_dc_voltages(self, dc_voltages, include_gaps=True):
        dc_initial = self.v_dc
        y0 = []
        for v_dc in dc_voltages:
            self.v_dc = v_dc
            y0.append(self.find_equilibrium_height(include_gaps=include_gaps))

        self.v_dc = dc_initial
        return np.array(y0)

    def plot_E_field(self, x_range=(-15E-3, 20E-3), y_range=(0.E-3, 10.E-3), resolution=(512, 512), include_gaps=True, v_dc = 0, v_ac = 1500):
        self.v_dc = v_dc
        self.v_rf = v_ac
        x = np.linspace(x_range[0], x_range[1], num=resolution[0])
        y = np.linspace(y_range[0], y_range[1], num=resolution[1])
        x, y = np.meshgrid(x, y)
        E_x, E_y = self.grad_phi_ac_gaps(x, y) if include_gaps else self.grad_phi_ac(x, y)
        phi_ac = self.phi_ac(x, y) if include_gaps else self.phi_ac_with_gaps(x, y)
        fig, ax = plt.subplots(1, 1)
        color = 'yellowgreen' if v_ac >= 0 else 'midnightblue'
        ax.streamplot(x, y, -E_x, -E_y, density=2, color=color)
        # ax.quiver(x, y, -E_x, -E_y)
        ax.pcolormesh(x, y, phi_ac, cmap='viridis')
        fig.tight_layout()
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        # plt.colorbar()
        plt.show()
        return
def plot_trap_escape_vary_dc(trap: PseudopotentialPlanarTrap, dc_values=np.linspace(0., 150., num=20), xrange=[-7.4E-3, 12.4E-3], xnum=100, include_gaps=True):
    fig, ax = plt.subplots(1, 1)
    xs = np.linspace(xrange[0], xrange[1], num=xnum)
    y0s = []
    colors = get_sequential_colormap(len(dc_values))
    for i, v_dc in enumerate(dc_values):
        trap.v_dc = v_dc
        y0s.append(trap.find_equilibrium_height(include_gaps))
        ax.plot(xs * 1.E3, trap.u_total(xs, y0s[-1], include_gaps), label=f'{v_dc:.1f} V', color=colors[i])
    ax.grid()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('potential energy / charge at equilibrium height (V)')
    fig.legend()
    figh, axh = plt.subplots(1, 1)
    axh.plot(dc_values, y0s, marker='o', linestyle='None')
    axh.set_xlabel('DC electrode voltage (V)')
    axh.set_ylabel('equilibrium height (mm)')
    plt.show()

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
    meas_min = rawdata[np.argmin(rawdata[:, 1])]
    #Rought guess based on chart. Need more precise measurement from Cole.
    return dc_voltages, y0 * 1.E-3, y_spread * 1.E-3, meas_min


def fit_data(trap: PseudopotentialPlanarTrap, parameters, bounds=None, include_gaps=False):
    dc_voltages, y0, yspread, meas_min = get_data()
    fig, ax = plt.subplots(1, 1)
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
    ax.plot(model_voltages, y0_model * 1.E3, label=r'Fitted Charge-to-mass: $\gamma_f =\ $' + str(np.round(trap.charge_to_mass * 10**4, decimals=3)) + r'$\times\ 10^{-4}\ C/kg$')
    trap.v_dc = meas_min[0]
    rf_null_meas = trap.grad_u_dc(trap.a / 2, meas_min[2]*10**-3, include_gaps=include_gaps)
    trap.charge_to_mass = -g/rf_null_meas
    print("c_t_m: " + str(trap.charge_to_mass))
    y0_meas = trap.get_height_versus_dc_voltages(model_voltages, include_gaps=include_gaps)
    plt.rcParams["text.usetex"]=True
    ax.plot(dc_voltages, y0 * 1.E3, marker='o', linestyle='None')
    plt.errorbar(dc_voltages, y0 * 1.E3, yerr=0.0164, fmt='none', ls='none', capsize=5)
    ax.plot(model_voltages, y0_meas * 1.E3, label=r'RF-null Charge-to-mass: $\gamma_n =\ $' + str(np.round(trap.charge_to_mass * 10**4, decimals=3)) + r'$\times\ 10^{-4}\ C/kg$')
    ax.set_xlabel('DC electrode voltage (V)')
    ax.set_ylabel('Ion height (mm)')
    ax.grid()
    ax.legend()
    fig.suptitle(f'include_gaps={include_gaps}')
    return trap

def compare_model_gaps_versus_no_gaps(trap: PseudopotentialPlanarTrap):
    fit_data(trap, ['charge_to_mass'], bounds=[(1.E-4, 1.E-2)], include_gaps=False)
    fit_data(trap, ['charge_to_mass'], bounds=[(1.E-4, 1.E-2)], include_gaps=True)

    fig, ax = trap.plot_y_cuts(include_gaps=True)
    fig.suptitle('including gaps')
    fig, ax = trap.plot_y_cuts(include_gaps=False)
    fig.suptitle('ignoring gaps')

    trap.plot_potential_at_surface()
    fig, ax = trap.plot_potential_contours(include_gaps=True, vmax=350)
    fig.suptitle('including gaps')
    fig, ax = trap.plot_potential_contours(include_gaps=False, resolution=(512, 512), vmax=350)
    fig.suptitle('excluding gaps')

    trap.plot_E_field(include_gaps=True, v_ac= 1000)
    trap.plot_E_field(include_gaps=True, v_ac=-1000)

    plt.show()


if __name__ == "__main__":
    trap = PseudopotentialPlanarTrap()
    trap.v_dc = 100
    compare_model_gaps_versus_no_gaps(trap)
    # plot_trap_escape_vary_dc(trap, include_gaps = True)
    get_data()
    # trap.plot_E_field(include_gaps=True, v_ac = -1000)
    # trap.plot_E_field(include_gaps=True, v_ac= -1000)