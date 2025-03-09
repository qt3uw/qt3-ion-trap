import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from dataclasses import dataclass, field


@dataclass
class Uncertainties:
    # Measured
    delta_endcap: float = 0.0
    delta_AC: float = 0.0
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
    delta_c2m_1: list[float]= -1e-3
    delta_c2m_2: list[float]= -1e-3

    
    
    
    def get_unc_values(self):
        return self.__dict__.copy()


    
    def set_values(self, **kwargs):
        for key, value in kwargs.iteritems():
             self.__dict__[key] = value

  
   
    def pxl_to_r(self):
        pxl_to_milli = np.array([(self.N_c[i]/ self.r_c[i]) for i in range(0, 2)]).tolist()
        self.pxl_to_mm = pxl_to_milli
        return pxl_to_milli
    
    def delta_pos_calc(self, r_sta = [[-1e-6], [-1e-6]]):
        pxl_to_vec = self.pxl_to_r
        def delta_r(est_diff_pairs, r_est):
            sum_i = []
            sum_tot = 0
            for summand in est_diff_pairs:
     
                sum_i.append(np.square((np.divide(summand[0], summand[1]))))
            print(sum_i)
            print(r_est)
            return [np.multiply(r_est[0], np.sqrt(np.sum(sum_i))), np.multiply(r_est[1], np.sqrt(np.sum(sum_i)))]


        est_diff_pairs = [[self.r_c, self.delta_r_c], [self.N_c, self.delta_N_c], [self.diff_N, self.delta_diff_N]]
        print(delta_r(est_diff_pairs, self.r))
        print((3*np.array(r_sta))[0].shape)
        self.delta_r = (delta_r(est_diff_pairs, self.r) + (3*np.array(r_sta))).tolist() 
        return (delta_r(est_diff_pairs, self.r) + (3*np.array(r_sta))).tolist() 

    
    def diff_N_calc(self):
        self.diff_N = np.diff(np.array([self.N_i, self.N_f]), axis=0)[0, :].tolist()
    
    
    def delta_t_calc(self):
        self.delta_t = self.T_exp / 2

    
    def delta_v_calc(self):
        self.pxl_to_r()
        self.delta_pos_calc()
        self.delta_v = np.abs(v) * np.sqrt((self.delta_r[0]/self.r[0])**2 + (self.delta_t / self.t)**2)

    
    def fit_error(self, y_fit, sigma, trap):
        V_inv = np.asmatrix(np.diag(np.square(sigma))).I
        F1 = trap.u_gravity(np.ones_like(y_fit) * trap.a / 2, y_fit)
        F2 = trap.u_ac(np.ones_like(y_fit) * trap.a / 2 * trap.a / 2, y_fit)
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