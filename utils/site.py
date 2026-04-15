import utils.constants as C
import numpy as np

class Site:
    def __init__(self, name, H, Z, sample_year, rho_0, use_HL=False): #C, m ieq/ yr, hPa, CE, m weq/yr
        self.name = name
        self.T_surf = None
        self.T_basal = None
        # self.rho_ice = rho_ice
        self.H = H
        self.Z = Z
        self.A_ieq = None
        self.p_atm = None
        self.C_atm = None
        self.sample_year = sample_year
        self.rho_0 = rho_0

        self.use_HL = use_HL

    def _sort_data(self, arr):
        sorted_indices = arr[:, 0].argsort()
        sorted_arr = arr[sorted_indices]
        return sorted_arr

    def read(self):
        self.A_ieq = np.loadtxt(C.ROOT + "icecores\\" + self.name + "\\accumulation_rate.txt")
        self.A_ieq = self._sort_data(self.A_ieq)

        self.T_surf = np.loadtxt(C.ROOT + "icecores\\" + self.name + "\\temperature_surface.txt")
        self.T_surf = self._sort_data(self.T_surf)

        self.T_basal = np.loadtxt(C.ROOT + "icecores\\" + self.name + "\\temperature_basal.txt")
        self.T_basal = self._sort_data(self.T_basal)
        # self.T_surf += C.C_to_K
        # self.T_basal += C.C_to_K
        self.p_atm = np.loadtxt(C.ROOT + "icecores\\" + self.name + "\\pressure_surface.txt")
        self.p_atm = self._sort_data(self.p_atm)
        self.p_atm[:, 1] *= C.hPa_to_Pa

        self.C_atm = np.loadtxt(C.ROOT + "icecores\\" + self.name + "\\concentration_surface.txt")
        self.C_atm = self._sort_data(self.C_atm)
        #if pressure unit is hPa
        
        # self.rho_ice = 0.9165 - self.T_surf * 1.4438E-4 - self.T_surf ** 2 * 1.5175E-7
        # self.A_weq = self.A_ieq.copy()

        # #TODO: rho ice 가변
        # self.A_weq[:, 1] *= self.rho_ice[0]
        return