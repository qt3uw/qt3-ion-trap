import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches

class Uncertainties:
     def __init__(self,exper):
         self.endcap_unc = 0.0
         self.AC_unc = 0.0
         self.freq_unc = 0.0
         self.ruler_error = 0.0
         if exper == 1:
             self.y_interp_error = 0.0 
             self.ruler_error = 0.0
             self.covar_matrix = 0.0
         else:
            self.x_interp_error = 0.0
            self.t_unc = 0.0
            self.t_exp = 0.0

     def get_unc_values(self):
        return self.__dict__.copy()