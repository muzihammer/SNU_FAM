import utils.constants as C

class Site:
    def __init__(self, name, T, A_ieq, P, H, sample_year, rho_0, rho_ice, use_HL=False): #C, m ieq/ yr, hPa, CE, m weq/yr
        self.name = name
        self.T = T + C.C_to_K
        self.rho_ice = rho_ice
        self.H = H
        self.A_ieq = A_ieq
        self.A_weq = self.A_ieq * self.rho_ice
        self.p_0 = P * C.hPa_to_Pa
        self.sample_year = sample_year

        self.rho_0 = rho_0

        self.use_HL = use_HL

    def read(self, name):

        return