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
        Calculates the free-space voltage from the AC electrode.
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

    def pseudo(self, x, y):
        gradx, grady = self.grad_phi(x, y)
        return (self.charge_to_mass / (4 * self.omega ** 2)) * np.sqrt(gradx ** 2 + grady ** 2)

    def phi_dc(self, x, y):
        return (self.v_dc / np.pi) * (np.arctan((self.x2 - x) / y) - np.arctan((self.x1 - x) / y))

    def v_gravity(self, x, y):
        return (1. / self.charge_to_mass) * g * y

    def charToMassCalc(x, y, vDc):
        raise NotImplementedError
        return g / ((vDc * (-((x2 - x) * 1 / 1000) / (
                    (((x2 - x) * 1 / 1000) ** 2 / (y * 1 / 1000) ** 2 + 1) * (y * 1 / 1000) ** 2) - (x * 1 / 1000) / (
                                        (((x * 1 / 1000) / (y * 1 / 1000) - (x1 * 1 / 1000)) ** 2 + 1) * (
                                            y * 1 / 1000) ** 2))) / Pi)

    def trapDepth(cm, vRf, a, b):
        raise NotImplementedError
        return (cm * vRf ** 2 / (Pi ** 2 * Omega ** 2)) * ((b * 0.001) / (
                    (a * 0.001 + b * 0.001) ** 2 + (a * 0.001 + b * 0.001) * np.sqrt(
                2 * a * b * 0.001 * 0.001 + (a * 0.001) ** 2))) ** 2

if __name__ == "__main__":
    p = PseudopotentialPlanarTrap()
    y = np.linspace(0.E-3, 10.E-3, num=500)
    x = np.zeros_like(y) + p.a / 2.

    v_grav = p.v_gravity(x, y)
    phi_dc = p.phi_dc(x, y)
    psuedo = p.pseudo(x, y) / p.charge_to_mass

    fig, ax = plt.subplots(1, 1)
    ax.plot(y * 1.E3, psuedo, label='psuedo')
    ax.plot(y * 1.E3, phi_dc, label='dc')
    ax.plot(y * 1.E3, v_grav, label='gravity')
    ax.plot(y * 1.E3, v_grav + phi_dc + psuedo, label='total')
    ax.set_xlabel('y (mm)')
    ax.grid()
    ax.set_ylabel('potential (V)')
    plt.show()
