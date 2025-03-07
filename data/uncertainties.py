import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import List

@dataclass
class Uncertainties:
    # Measured
    delta_endcapc: float = 0.0
    delta_AC: float = 0.0
    delta_freq: float = 0.0
    delta_ruler: float = 0.05 * 25.4
    T_exp: float = 1
    v: float = 1
    t: float = 1
    pxl_to_mm:List =  field(default_factory= lambda: [0.006, 0.006])
    
    r_c: List = field(default_factory= lambda: [25.4 * 1/64, 25.4 * 1/64])
    delta_r_c: List = field(default_factory= lambda: [25.4* 1/64, 25.4* 1/64])
    N_c: List = field(default_factory= lambda: [1, 1])
    delta_N_c: List = field(default_factory= lambda: [1, 1])
    delta_t: List = field(default_factory= lambda: [1, 1])
    N_f:  List = field(default_factory= lambda: [2, 2])
    N_i:  List = field(default_factory= lambda: [1, 1])
    delta_N_f: List = field(default_factory= lambda: [2, 2])
    delta_N_i:  List = field(default_factory= lambda: [1, 1])
    delta_diff_N: List =  field(default_factory= lambda: [3/(2*np.sqrt(6)), 3/(2*np.sqrt(6))])
    # Calculated
    r: List =  field(default_factory= lambda: [3.5, 1], init=False)
    delta_r: List = field(default_factory= lambda: [25.4* 1/64, 25.4 * 1/64], init=False)
    diff_N: List = field(default_factory= lambda: [1, 1], init=False)
    delta_v: List = field(default_factory= lambda: [1, 1], init=False)
    delta_c2m_1: float = -1e-3
    delta_c2m_2: float = -1e-3

    
    
    @property
    def get_unc_values(self):
        return self.__dict__.copy()


    @property
    def set_values(self, **kwargs):
        for key, value in kwargs.iteritems():
             self.__dict__[key] = value

  
    @property
    def pxl_to_r(self):
        self.pxl_to_mm = np.array([(self.r_c[i] / self.N_c[i]) for i in range(0, 2)]).tolist()
        self.r = [self.pxl_to_mm[i] * (self.N_f[i] - self.N_i[i]) for i in range(0, 2)]

    @property
    def delta_pos_calc(self, r_std = [-1e-6, -1e-6]):
        self.pxl_to_r()
        def delta_r(est_diff_pairs, r_est):
            sum_i = []
            sum_tot = 0
            for summand in est_diff_pairs:
     
                sum_i.append(np.square((np.divide(summand[0], summand[1]))))
            print(sum_i)
            print(r_est)
            return np.multiply(r_est, np.sum(sum_i))


        est_diff_pairs = [[self.r_c, self.delta_r_c], [self.N_c, self.delta_N_c], [self.diff_N, self.delta_diff_N]]
        self.delta_r = (delta_r(est_diff_pairs, self.r) + np.array(r_std)).tolist() 

    @property
    def diff_N_calc(self):
        self.diff_N = np.diff(np.array([self.N_i, self.N_f]), axis=0)[0, :].tolist()
    
    @property
    def delta_t_calc(self):
        self.delta_t = self.T_exp / 2

    @property
    def delta_v_calc(self):
        self.pxl_to_r()
        self.delta_pos_calc()
        self.delta_v = np.abs(v) * np.sqrt((self.delta_r[0]/self.r[0])**2 + (self.delta_t / self.t)**2)

if __name__ == "__main__":
    U = Uncertainties()

    print(U.diff_N_calc)
    print(U.delta_r)
    print(U.N_f)
    print(U.N_i)
    print(U.diff_N)
