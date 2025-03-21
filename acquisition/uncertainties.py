import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from dataclasses import dataclass, field


@dataclass
class Uncertainties:
    """
    Used to store uncertainty information related to 
    measured values in experiments. Also contains methods to derive 
    uncertainty values for derived physical properties of the 
    system of interest. 
        :attr delta_endcap: (float) Uncertainty value for endcap voltage
        :attr delta_AC: (float) Uncertainty value for AC electrode voltage
        :attr delta_cent: (float) Uncertainty value for central electrode voltage
        :attr delta_freq: (float) Uncertainty value in the AC frequency
        :attr delta_ruler: (float) Uncertainty value for the calibration ruler
        :attr T_exp: (float) Exposure time furing experiment
        :attr v: (float) velocity of a particle 
        :attr t: (float) time value(s) in seconds
        :attr pxl_to_mm: (List[float, float]) A list of [pixel] / [mm] values for the x-axis and y-axis respectfully.
        :attr r_c: (List[float, float]) A list of calibration distances measured with the tuler calibration image
                    for the x-axis and y-axis respectfully
        :attr delta_r_c: (List[float, float]) A list of calibration-distance uncertainties for the x-axis and y-axis
                         respectfully
        :attr N_c: (List[float, float]) A list of calibrtion distances in units of pixels for the x-axis and y-axis
                   respectfully
        :attr delta_N_c: (List[float, float]) A list of calibration-distance undertainties in pixels for the x-axis and
                         y-axis respectfully
        :attr delta_t: (List[float, float]) Uncertainty in the time value in seconds
        :attr N_f: (List[float, float]) Location of particle in units of pixels along the x-axis and y-axis
                   respectfully
        :attr N_i: (List[float, float])  Location of planar trap surface in units of pixels along the x-axis and y-axis
                   respectfully
        :attr delta_N_f: (List[float, float]) Unceratinty values in N_f in units of pixels along the x-axis and y-axis
                         respectfully
        :attr delta_N_i: (List[float, float]) Uncertainty values in N_i in units of pixels along the x-axis and y-axis 
                         respectfully
        :attr delta_diff_N: (List[float, float]) Uncertainty in N_f - N_i along the x-axis and y-axis respectfully
        :attr r: (List[float, float]) Position of particle in units of millimeters along the x-axis and y-axis 
                 respectfully
        :attr delta_r: (List[float, float]) Uncertainty values in 'r' along the x-axis and y-axis respectfully
        :attr diff_N: (List[float, float]) N_f - N_i along the x-axis and y-axis respectfully
        :attr delta_v: (List[float, float]) Uncertainty values in 'v' along the x-axis and y-axis respectfully
        :attr delta_c2m_1: (float) Uncertainty in the charge-to-mass ratio calculated with method 1 in [Coulombs] / [kilogram]
        :attr delta_c2m_2: (float) Uncertainty in the charge-to-mass ratio calculated with method 2 in [Coulombs] / [kilogram]
    """
    # Measured
    delta_endcap: float = 0.0
    delta_AC: float = 0.0
    delta_cent: float = -1.0
    delta_freq: float = 0.0
    delta_ruler: float = 0.05 * 25.4
    T_exp: float = 1
    v: float = 1
    t: float = 1
    pxl_to_mm: list[float] =  field(default_factory= lambda: [0.006, 0.006])
    r_c: list[float] = field(default_factory= lambda: [25.4 * 1/64, 25.4 * 1/64])
    delta_r_c: list[float] = field(default_factory= lambda: [25.4* 1/64, 25.4* 1/64])
    N_c: list[float]= field(default_factory= lambda: [1, 1])
    delta_N_c: list[float] = field(default_factory= lambda: [1, 1])
    delta_t: list[float]= field(default_factory= lambda: [1, 1])
    N_f: list[float]= field(default_factory= lambda: [1, 1])
    N_i: list[float]= field(default_factory= lambda: [1, 1])
    delta_N_f: list[float] = field(default_factory= lambda: [1, 1])
    delta_N_i: list[float]= field(default_factory= lambda: [1, 1])
    delta_diff_N: list[float]=  field(default_factory= lambda: [3/(2*np.sqrt(6)), 3/(2*np.sqrt(6))])
    # Calculated
    r: list[float] =  field(default_factory= lambda: [0, 0], init=False)
    delta_r: list[float] = field(default_factory= lambda: [25.4* 1/64, 25.4 * 1/64], init=False)
    diff_N: list[float]= field(default_factory= lambda: [1, 1], init=False)
    delta_v: list[float] = field(default_factory= lambda: [1, 1], init=False)
    delta_c2m_1: float = -1e-3
    delta_c2m_2: float = -1e-3

    
    
    
    def get_unc_values(self):
        """
        Returns copy of uncertainties dictionary
        """
        return self.__dict__.copy()
    
    def set_values(self, **kwargs):
        """
        Assign values to fields for an Uncertainties() object
        """
        for key, value in kwargs.iteritems():
             self.__dict__[key] = value

    def pxl_to_r(self):
        """
        Calculates, saves in a field, and returns the pixel-to-millimeter conversion 
        factor.
            :return pxl_to_milli: returns pixel-to-millimeter conversion factor 
        """
        pxl_to_milli = np.array([(self.N_c[i]/ self.r_c[i]) for i in range(0, 2)]).tolist()
        self.pxl_to_mm = pxl_to_milli
        return pxl_to_milli
    
    def delta_pos_calc(self, r_sta = [[1e-6], [1e-6]]):
        """
        Calculates uncertainty values for the particle's position along the x-axis and y-axis
        respectfully
            :returns: uncertainty values for the particle's position along the x-axis and y-axis
                      respectfully 
        """
        pxl_to_vec = self.pxl_to_r
        def delta_r(est_diff_pairs, r_est):
            sum_i = [[], []]
            sum_tot = 0
            for i in range(2):
                sum_i[i].append(np.square((np.divide(est_diff_pairs[i][1], est_diff_pairs[i][0]))))
            return [np.multiply(r_est[i], np.sqrt(np.sum(sum_i, axis=0))[0][i]) for i in range(2)]
        est_diff_pairs = [[self.r_c, self.delta_r_c], [self.N_c, self.delta_N_c]]
        return (delta_r(est_diff_pairs, self.r) + (3*np.array(r_sta))).tolist() 

    
    def diff_N_calc(self):
        """
        Calculates and assigns the corresponging class property the difference between 
        pixel values for the particle height and the planar trap surface.
        """
        self.diff_N = np.diff(np.array([self.N_f, self.N_i]), axis=0)[0, :].tolist()
    
    
    def delta_t_calc(self):
        """
        Calculates the uncertainty in time of an event in the video
            :returns: NotImplementedError
        """
        #self.delta_t = self.T_exp / 2
        raise NotImplementedError

    
    def delta_v_calc(self):
        """
        Calculates the uncertainty in velocity of the particle
            :returns: NotImplementedError
        """
        # self.pxl_to_r()
        # self.delta_pos_calc()
        # self.delta_v = np.abs(v) * np.sqrt((self.delta_r[0]/self.r[0])**2 + (self.delta_t / self.t)**2)
        raise NotImplementedError

    
def fit_error(y_fit, sigma, trap):
    """
    Fits the minimum potential values along the y-axis 
    as a function of the charge-to-mass value coefficients
        :returns U: The Covarient matrix of the two 
        optimized parameters, which are 1/charge_to_mass
        and charge_to_mass.
    """
    V_inv = np.asmatrix(np.diag(np.square(sigma))).I
    F1 = trap.u_gravity(np.ones_like(y_fit) * trap.a / 2, y_fit)
    F2 = trap.u_ac(np.ones_like(y_fit) * trap.a / 2, y_fit)
    #F3 = [trap.u_dc(trap.a / 2, y_i) for y_i in y_fit]
    F = np.asmatrix([F1, F2])
    U =  (F @ V_inv @ F.T).I
    return U

"""
if __name__ == "__main__":
    U = Uncertainties()
    print(U.diff_N_calc)
    print(U.delta_r)
    print(U.N_f)
    print(U.N_i)
    print(U.diff_N)
"""