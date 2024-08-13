from dataclasses import dataclass

from scipy.constants import g
import matplotlib.pyplot as plt
import numpy as np

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
        return self.ac_electrode_width + self.ac_electrode_inner_gap

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
        return self.v_rf / np.pi * (
                    np.arctan((self.a + self.b - x) / y) - np.arctan((self.a - x) / y) -
                    np.arctan(x / y) + np.arctan((self.c + x) / y))

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
        return (1. / (4 * self.omega ** 2)) * np.sqrt(gradx ** 2 + grady ** 2)

    def u_dc(self, x, y):
        """
        Returns potential from the DC electrodes, divided by the charge to mass ratio

        :param x:
        :param y:
        :return:
        """
        return (self.v_dc / np.pi) * (np.arctan((self.x2 - x) / y) - np.arctan((self.x1 - x) / y))

    def u_gravity(self, x, y):
        """
        Returns gravitational potential, divided by the charge to mass ratio

        :param x:
        :param y:
        :return:
        """
        return (1. / self.charge_to_mass) * g * y

    def u_total(self, x, y):
        "Returns sum of gravitational, dc, and ac pseudopotential divided by charge to mass ratio"
        return self.u_gravity(x, y) + self.u_dc(x, y) + self.u_ac(x, y)

    def plot_potential_contours(self, x_range=[0., 10.E-3], y_range=[0., 10.E-3], resolution=[512, 512],
                                fig=None, ax=None):
        x = np.linspace(x_range[0], x_range[1], num=resolution[0])
        y = np.linspace(y_range[0], y_range[1], num=resolution[1])
        x, y = np.meshgrid(x, y)
        u = self.u_total(x, y)
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        if fig is None:
            fig, ax = plt.subplots(1, 1)
            levels = np.linspace(np.min(u.flatten()), np.max(u.flatten()), num=20)
            # ax.contour(cs, colors='k')
            cs = ax.imshow(np.flipud(u), extent=extent)
            ax.contour(u, levels=15, colors='k', extent=extent)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('height (mm)')
            ax.set_title('total trap potential energy (J / C / kg)')
            fig.colorbar(cs)


if __name__ == "__main__":
    p = PseudopotentialPlanarTrap()
    p.v_dc = 0.
    p.plot_potential_contours()
    plt.show()
    # p.v_dc = 10.
    # y = np.linspace(0.01E-3, 10.E-3, num=500)
    # x = np.zeros_like(y) + p.a / 2.
    #
    # v_grav = p.u_gravity(x, y)
    # phi_dc = p.u_dc(x, y)
    # psuedo = p.u_ac(x, y)
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(y * 1.E3, psuedo, label='psuedo')
    # ax.plot(y * 1.E3, phi_dc, label='dc')
    # ax.plot(y * 1.E3, v_grav, label='gravity')
    # ax.plot(y * 1.E3, v_grav + phi_dc + psuedo, label='total')
    # fig.legend()
    # ax.set_xlabel('y (mm)')
    # ax.grid()
    # ax.set_ylabel('potential (J / (coulomb / kg)')
    # plt.show()
    #
    # figim, axim = plt.subplots(1, 1)
