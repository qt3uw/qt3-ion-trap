import copy
from dataclasses import dataclass

import scipy.optimize
from scipy.constants import g
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from numpy import linalg as LA

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

def get_sequential_colormap(num, cmap='viridis', cmin=0.0, cmax=1.0):
    xs = np.linspace(cmin, cmax, num=num)
    cs = mpl.colormaps[cmap]
    return [cs(x) for x in xs]

@dataclass
class PseudopotentialPlanarTrap:
    central_electrode_width: float = 3.175E-3
    ac_electrode_width: float = 4.15831E-3
    v_rf: float = 0.5 * 50 * 75
    v_dc: float = -200.
    charge_to_mass: float = 6.8E-4
    freq_rf: float = 60.
    gap_width: float = 2.E-3
    shuttle_width: float = 16.491E-3
    electrode_height: float = .5E-3
    v_error: float = 0.0164E-3

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
        return -self.v_dc * ((0.001 - x) / ((1 + (0.001 - x) ** 2 / y ** 2) * y ** 2) - (0.004175 - x) / ((1 + (0.004175 - x) ** 2 / y ** 2) * y ** 2)) / np.pi
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

    def find_equilibrium_height(self, ystep=1.E-6, guess=2.5E-3, include_gaps=True):
        def merit_func(y):
            ys = np.linspace(y-ystep, y+ystep, num=3)
            xs = np.zeros_like(ys) + self.a / 2.
            return -self.u_total(xs, ys, include_gaps=include_gaps).flatten()[1]
        res = minimize_scalar(merit_func, bounds=(0.5E-3, 20.E-3))
        return res.x

    def get_height_versus_dc_voltages(self, dc_voltages, include_gaps=True):
        dc_initial = self.v_dc
        y0 = []
        for v_dc in dc_voltages:
            self.v_dc = v_dc
            y0.append(self.find_equilibrium_height(include_gaps=include_gaps))

        self.v_dc = dc_initial
        return np.array(y0)

    def draw_electrodes(self, ax, include_gaps=True):
        hatch = '..'
        hatch2 = '//'
        facecolor = 'wheat'
        edgecolor = 'tan'
        electrode_height = self.electrode_height
        dielec_height = .5E-3
        dielec_facecolor = 'rebeccapurple'
        dielec_edgecolor = 'indigo'
        if include_gaps:
            ax.add_patch(
                Rectangle((self.gap_width / 2, 0), self.central_electrode_width, -electrode_height, facecolor=facecolor,
                          edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(Rectangle((self.a - self.gap_width / 2, 0), self.gap_width, -electrode_height,
                                   facecolor=dielec_facecolor, edgecolor=dielec_edgecolor, hatch=hatch))
            ax.add_patch(
                Rectangle((- self.gap_width / 2, 0), self.gap_width, -electrode_height, facecolor=dielec_facecolor,
                          edgecolor=dielec_edgecolor, hatch=hatch))
            ax.add_patch(Rectangle((self.a + self.gap_width / 2, 0), self.ac_electrode_width, -electrode_height,
                                   facecolor=facecolor, edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(Rectangle((self.a + self.b - self.gap_width / 2, 0), self.gap_width, -electrode_height,
                                   facecolor=dielec_facecolor, edgecolor=dielec_edgecolor, hatch=hatch))
            ax.add_patch(Rectangle((-self.ac_electrode_width - self.gap_width / 2, 0), self.ac_electrode_width,
                                   -electrode_height, facecolor=facecolor,
                                   edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(Rectangle((-self.c - self.gap_width / 2, 0), self.gap_width,
                                   -electrode_height, facecolor=dielec_facecolor,
                                   edgecolor=dielec_edgecolor, hatch=hatch))
            ax.add_patch(
                Rectangle((-self.c - self.gap_width / 2 - self.shuttle_width, 0), self.shuttle_width, -electrode_height,
                          facecolor=facecolor, edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(
                Rectangle((self.a + self.b + self.gap_width / 2, 0), self.shuttle_width, -electrode_height,
                          facecolor=facecolor, edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(Rectangle((-self.c - self.gap_width / 2 - self.shuttle_width, -electrode_height / 2), -(
                        -self.c - self.gap_width / 2 - self.shuttle_width) + self.a + self.b + self.gap_width / 2 + self.shuttle_width,
                                   -electrode_height / 2, facecolor=dielec_facecolor,
                                   edgecolor=dielec_edgecolor, hatch=hatch))
            ax.add_patch(Rectangle((-self.c - self.gap_width / 2 - self.shuttle_width, -electrode_height / 2), -(
                    -self.c - self.gap_width / 2 - self.shuttle_width) + self.a + self.b + self.gap_width / 2 + self.shuttle_width,
                                   -electrode_height / 2, facecolor='none',
                                   edgecolor=dielec_facecolor))
        else:
            ax.add_patch(
                Rectangle((0, 0), self.a, -electrode_height, facecolor=facecolor, edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(Rectangle((self.a, 0), self.b, -electrode_height, facecolor=facecolor, edgecolor=edgecolor,
                                   hatch=hatch2))
            ax.add_patch(Rectangle((-self.c, 0), self.c, -electrode_height, facecolor=facecolor, edgecolor=edgecolor,
                                   hatch=hatch2))
            ax.add_patch(
                Rectangle((-self.shuttle_width - self.c, 0), self.shuttle_width, -electrode_height, facecolor=facecolor,
                          edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(Rectangle((self.a + self.b, 0), self.shuttle_width, -electrode_height, facecolor=facecolor,
                                   edgecolor=edgecolor, hatch=hatch2))
            ax.add_patch(Rectangle((-self.c - self.gap_width / 2 - self.shuttle_width, -electrode_height / 2), -(
                    -self.c - self.gap_width / 2 - self.shuttle_width) + self.a + self.b + self.gap_width / 2 + self.shuttle_width,
                                   -electrode_height / 2, facecolor=dielec_facecolor,
                                   edgecolor=dielec_edgecolor, hatch=hatch))
            ax.add_patch(Rectangle((-self.c - self.gap_width / 2 - self.shuttle_width, -electrode_height / 2), -(
                    -self.c - self.gap_width / 2 - self.shuttle_width) + self.a + self.b + self.gap_width / 2 + self.shuttle_width,
                                   -electrode_height / 2, facecolor='none',
                                   edgecolor=dielec_facecolor))
        return ax

    def plot_E_field(self, x_range=(-15E-3, 20E-3), y_range=(0.E-3, 10.E-3), resolution=(512, 512), include_gaps=True,
                     normalized=True, figsize=(10, 6)):

        x = np.linspace(x_range[0], x_range[1], num=resolution[0])
        y = np.linspace(y_range[0], y_range[1], num=resolution[1])
        x, y = np.meshgrid(x, y)

        E_x0, E_y0 = self.grad_phi_ac_gaps(x, y) if include_gaps else self.grad_phi_ac(x, y)
        if normalized:
            E_x0 = E_x0 / np.sqrt(E_x0 ** 2 + E_y0 ** 2)
            E_y0 = E_y0 / np.sqrt(E_x0 ** 2 + E_y0 ** 2)
        phi_ac0 = self.phi_ac(x, y) if include_gaps == False else self.phi_ac_with_gaps(x, y)

        self.v_rf = -self.v_rf
        E_x1, E_y1 = self.grad_phi_ac_gaps(x, y) if include_gaps else self.grad_phi_ac(x, y)
        if normalized:
            E_x1 = E_x1 / np.sqrt(E_x1 ** 2 + E_y1 ** 2)
            E_y1 = E_y1 / np.sqrt(E_x1 ** 2 + E_y1 ** 2)
        phi_ac1 = self.phi_ac(x, y) if include_gaps == False else self.phi_ac_with_gaps(x, y)

        fig, ax = plt.subplots(1, 2, figsize=figsize, layout="compressed")

        color = 'yellowgreen'
        ax[0].streamplot(x, y, -E_x0, -E_y0, density=(.75, .75), color=color, arrowstyle='fancy', linewidth=1.25,
                         arrowsize=1)
        ax[1].streamplot(x, y, -E_x1, -E_y1, density=(.75, .75), color=color, arrowstyle='fancy', linewidth=1.25,
                         arrowsize=1)
        cax = ax[0].pcolormesh(x, y, phi_ac0, cmap='twilight_r', vmin=np.minimum(-self.v_rf, self.v_rf),
                               vmax=np.maximum(-self.v_rf, self.v_rf))
        ax[1].pcolormesh(x, y, phi_ac1, cmap='twilight_r', vmin=np.minimum(-self.v_rf, self.v_rf),
                                vmax=np.maximum(-self.v_rf, self.v_rf))
        self.draw_electrodes(ax[0], include_gaps=include_gaps)
        self.draw_electrodes(ax[1], include_gaps=include_gaps)
        ax[0].set_xlim(x_range[0], x_range[1])
        ax[1].set_xlim(x_range[0], x_range[1])
        ax[0].set_ylim(y_range[0] - 0.5E-3, y_range[1])
        ax[1].set_ylim(y_range[0] - 0.5E-3, y_range[1])

        for a in ax:
            xticks = a.get_xticks()
            yticks = a.get_yticks()
            a.set_xticklabels([f'{tick * 1000:.0f}' for tick in xticks])
            a.set_yticklabels([f'{tick * 1000:.0f}' for tick in yticks])

        plt.setp(ax[1].get_yticklabels(), visible=False)

        inset_x = [self.a / 2 - .0015, self.a / 2 + .0015]
        inset_y = [0.00375 - .001, 0.00575 + .0015]
        ax_inset_0 = ax[0].inset_axes([.015, .45, .35, .5], xlim=[inset_x[0], inset_x[1]],
                                      ylim=[inset_y[0], inset_y[1]], xticklabels=[], yticklabels=[])
        ax_inset_1 = ax[1].inset_axes([.015, .45, .35, .5], xlim=[inset_x[0], inset_x[1]],
                                      ylim=[inset_y[0], inset_y[1]], xticklabels=[],
                                      yticklabels=[])
        ax[0].indicate_inset_zoom(ax_inset_0, edgecolor="yellow")
        ax[1].indicate_inset_zoom(ax_inset_1, edgecolor="yellow")
        ax_inset_0.tick_params(color='yellow', labelcolor='yellow')
        ax_inset_1.tick_params(color='yellow', labelcolor='yellow')
        x_inset = np.linspace(inset_x[0], inset_x[1], num=15)
        y_inset = np.linspace(inset_y[0], inset_y[1], num=8)
        x_inset, y_inset = np.meshgrid(x_inset, y_inset)
        self.v_rf = -self.v_rf
        E_x_inset0, E_y_inset0 = self.grad_phi_ac_gaps(x_inset, y_inset) if include_gaps else self.grad_phi_ac(x_inset,
                                                                                                               y_inset)
        if normalized:
            E_x_inset0 = E_x_inset0 / np.sqrt(E_x_inset0 ** 2 + E_y_inset0 ** 2)
            E_y_inset0 = E_y_inset0 / np.sqrt(E_x_inset0 ** 2 + E_y_inset0 ** 2)
        self.v_rf = -self.v_rf
        E_x_inset1, E_y_inset1 = self.grad_phi_ac_gaps(x_inset, y_inset) if include_gaps else self.grad_phi_ac(x_inset,
                                                                                                               y_inset)
        if normalized:
            E_x_inset1 = E_x_inset1 / np.sqrt(E_x_inset1 ** 2 + E_y_inset1 ** 2)
            E_y_inset1 = E_y_inset1 / np.sqrt(E_x_inset1 ** 2 + E_y_inset1 ** 2)

        self.v_rf = -self.v_rf
        ax_inset_0.pcolormesh(x, y, phi_ac0, cmap='twilight_r', vmin=np.minimum(-self.v_rf, self.v_rf),
                                      vmax=np.maximum(-self.v_rf, self.v_rf))
        ax_inset_0.quiver(x_inset, y_inset, -E_x_inset0, -E_y_inset0, color=color, scale=90000000, scale_units='x',
                          width=0.011)

        self.v_rf = -self.v_rf
        cax_2 = ax_inset_1.pcolormesh(x, y, phi_ac1, cmap='twilight_r', vmin=np.minimum(-self.v_rf, self.v_rf),
                                      vmax=np.maximum(-self.v_rf, self.v_rf))
        ax_inset_1.quiver(x_inset, y_inset, -E_x_inset1, -E_y_inset1, color=color, pivot='tip', scale=90000000,
                          scale_units='x', width=0.011)
        fig.colorbar(mappable=cax, location='top', ax=ax)
        return fig, ax

    def plot_rf_potential_contours(self, x_range=(-15E-3, 20E-3), y_range=(0.E-3, 10.E-3), resolution=(512, 512), include_gaps=True,
                                fig=None, ax=None, ncountours=25, min_contour_level=-20., figsize=(3.5, 3)):
        x = np.linspace(x_range[0], x_range[1], num=resolution[0])
        y = np.linspace(y_range[0], y_range[1], num=resolution[1])
        x, y = np.meshgrid(x, y)
        u = self.u_ac(x, y, include_gaps=include_gaps)
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        levels = np.linspace(min_contour_level, 0., num=ncountours)
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('height (mm)')
        ax.set_title(f'potential energy / charge (V)\n center electrode at {self.v_dc:.1f} V')
        cax = ax.pcolormesh(x, y, u, cmap='viridis', rasterized=True, vmin=min_contour_level)
        self.draw_electrodes(ax, include_gaps=include_gaps)
        ax.contour(u, levels=levels, colors='k', extent=extent, linewidths=0.75, linestyles='solid')
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0] - 0.5E-3, y_range[1])
        plt.colorbar(cax)
        return fig, ax

    def plot_y_cuts(self, yrange=(1.E-3, 10.E-3), num=200, x0=None, include_gaps=True, figsize=None):
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
        linewidth = 1.5
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # plt.yscale('log')
        ax.plot(y * 1.E3, -u_ac, label='pseudo', color = 'indigo',linewidth=linewidth)
        ax.plot(y * 1.E3, -u_ac * 50, label='pseudo x 50', color = 'indigo', linestyle='--',linewidth=linewidth)
        ax.plot(y * 1.E3, -u_dc, label='DC',  color='indianred',linewidth=linewidth)
        ax.plot(y * 1.E3, -u_grav, label='gravity', color='yellowgreen',linewidth=linewidth)
        ax.plot(y * 1.E3, -(u_ac + u_dc + u_grav), label='total', color = 'teal', linestyle='dashdot',linewidth=linewidth)
        ax.set_ylim([-10, -np.min(u_ac + u_dc + u_grav)])
        fig.legend()
        ax.set_xlabel('y (mm)')
        ax.grid(True)
        ax.set_ylabel('-potential energy / charge (J/C)')
        # fig.savefig('figures/Robert Figures/y_cuts.pdf')
        return fig, ax

def plot_trap_escape_vary_dc(trap: PseudopotentialPlanarTrap, dc_values=np.linspace(0., 320., num=16),
                             xrange=[-7.4E-3, 12.4E-3], xnum=512, include_gaps=True, ystep=1.E-6, figsize=(3.5, 3)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xs = np.linspace(xrange[0], xrange[1], num=xnum)
    y0s = []
    colors = get_sequential_colormap(len(dc_values))
    for i, v_dc in enumerate(dc_values):
        trap.v_dc = v_dc
        y0s.append(trap.find_equilibrium_height(include_gaps=include_gaps))
        y_lcl = np.linspace(y0s[-1] - ystep, y0s[-1] + ystep, num=3)
        x_lcl, y_lcl = np.meshgrid(xs, y_lcl)
        u = trap.u_total(x_lcl, y_lcl, include_gaps=include_gaps)
        ax.plot(x_lcl[1, :] * 1.E3, u[1, :], label=f'{v_dc:.0f} V', color=colors[i])
    ax.grid(True)
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('Potential energy / charge (J/C)')
    fig.legend(bbox_to_anchor=(1, 1), fontsize=10)
    # figh, axh = plt.subplots(1, 1)
    # axh.plot(dc_values, y0s, marker='o', linestyle='None')
    # axh.set_xlabel('DC electrode voltage (V)')
    # axh.set_ylabel('equilibrium height (mm)')
    # plt.show()
    return fig, ax


def compare_model_gaps_versus_no_gaps(trap: PseudopotentialPlanarTrap):
    # trap_copy = copy.deepcopy(trap)
    # fit_data(trap_copy, ['charge_to_mass'], bounds=[(1.E-4, 1.E-2)], include_gaps=False)
    # fit_data(trap_copy, ['charge_to_mass'], bounds=[(1.E-4, 1.E-2)], include_gaps=True)

    fig, ax = trap.plot_y_cuts(include_gaps=True)
    fig.suptitle('including gaps')
    fig, ax = trap.plot_y_cuts(include_gaps=False)
    fig.suptitle('ignoring gaps')

    # trap.plot_potential_at_surface()
    # fig, ax = trap.plot_potential_contours(include_gaps=True, vmax=350)
    # fig.suptitle('including gaps')
    # fig, ax = trap.plot_potential_contours(include_gaps=False, resolution=(512, 512), vmax=350)
    # fig.suptitle('excluding gaps')
    # trap.plot_E_field(include_gaps=True, v_ac=1000)
    # trap.plot_E_field(include_gaps=True, v_ac=-1000)

    plt.show()


if __name__ == "__main__":
    trap = PseudopotentialPlanarTrap()
    # compare_model_gaps_versus_no_gaps(trap)
    # plot_trap_escape_vary_dc(trap, include_gaps=True)
    # get_data()
    # trap.v_rf = 1000.
    # trap.plot_E_field(include_gaps=True)
    # trap.v_rf = -1000.
    # trap.plot_E_field(include_gaps=True)
    # trap.plot_potential_contours()
    plt.show()
