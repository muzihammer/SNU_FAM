import utils.constants as C

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class Profile:
    
    def __init__(self, z, site, gas):
        self.z = z
        self.N = z.shape[0]
        self.T = C.Time
        self.S = site

        self.M_X = gas.M
        self.lambda_X = 0

        # self.D_CO2_0 = 1.39E-5 * (self.S.T/C.C_to_K) ** 1.75 * (C.P0 / self.S.p_0)
        self.D_CO2_0 = 5.75E-10 * self.S.T ** 1.81 * (C.P0 / self.S.p_0)
        self.D_X_0 = gas.gamma_X * self.D_CO2_0

        self.Delta_z = C.dz
        self.Delta_t = C.dt * C.year_to_sec #년을 초로

        #######Herron and Langway, 1980#########
        
        self.rho = self._rho()
        ##############Mitchell et al., 2015##############
        self.rho_COD = 1 / (1 - 1 / 75) / (1 / C.kg_to_g / self.S.rho_ice + 7.02E-7 * self.S.T - 4.5E-5) / C.kg_to_g 
        # self.rho_COD = 75 / 74 / (1 / self.S.rho_ice + self.S.T * 6.95E-4 - 4.3E-2)
        self.rho_COD_bar = 1 / (1 / self.S.rho_ice + 6.95E-4 * self.S.T - 4.3E-2)    #Martinerie et al., 1994, Mitchell et al., 2015
        self.COD_idx = np.argmin(np.abs(self.rho - self.rho_COD))
        self.rho_COD = self.rho[self.COD_idx]
        self.z_COD = self.z[self.COD_idx]

        # self.M = np.argmin(np.abs(self.z - (self.z_COD + 10)))
        self.M = np.argmin(np.abs(self.z - C.Z))

        # self.rho_LID = self.rho_COD - 0.014                             #Blunier et al., 2000
        # self.LID_idx = np.argmin(np.abs(self.rho - self.rho_LID))


        self.s = self._s()
        self.s_cl = self._s_cl()
        self.s_op = self._s_op()

        ################OSU Model########################
        self.tau_inv_DZ = self._tau_inv_DZ()
        self.tau_inv_LIZ = self._tau_inv_LIZ()

        self.D_X = self._D_X()
        self.D_eddy = self._D_eddy()

        self.s_op_star = self._s_op_star()
        print("rho_LID : ", self.rho_LID, " rho_COD : ", self.rho_COD)
        print("LID : ", self.z_LID, "m, COD : ", self.z_COD, "m")

        ##############Buizert et al., 2011##############
        self.w_ice = self._w_ice()
        self.p_cl = self._p_cl()
        self.w_air = self._w_air()


        ################CIC Model#########################
        self.x_air = self._x_air()
        print("x_air : ", self.x_air * 1E3, "ml/kg")
        self.C_i_n = self._C_i_n()

        # self.Gamma, self.Delta = self._Delta()
        # self.median, self.FWHM = self._FWHM()



    def _rho(self):
        if self.S.use_HL:
            #######Herron and Langway, 1980#########
            k_0 = 11 * np.exp(-10160 / C.R / self.S.T)
            k_1 = 575 * np.exp(-21400 / C.R / self.S.T)
            z_crit = 1 / k_0 / self.S.rho_ice * (np.log(0.55 / (self.S.rho_ice - 0.55)) - np.log(self.S.rho_0 / (self.S.rho_ice - self.S.rho_0)))
            print("z_0.55 : ", z_crit)
            Z_0 = np.exp(self.S.rho_ice * k_0 * self.z + np.log(self.S.rho_0 / (self.S.rho_ice - self.S.rho_0)))
            rho_shallow = self.S.rho_ice * Z_0 / (1 + Z_0)

            Z_1 = np.exp(self.S.rho_ice * k_1 * (self.z - z_crit) / self.S.A_weq ** 0.5 + np.log(0.55 / (self.S.rho_ice - 0.55)))
            rho_deep = self.S.rho_ice * Z_1 / (1 + Z_1)

            z_crit_idx = np.argmin(np.abs(rho_deep - rho_shallow))

            return np.concatenate((rho_shallow[:z_crit_idx], rho_deep[z_crit_idx:]))
        else:
            rho = np.loadtxt(C.ROOT+"icecores\\"+self.S.name+"\\density_smoothed.txt")
            rho = np.interp(self.z, rho[:, 0], rho[:, 1])
            return rho
        # elif self.S.name == "NEEM_EU":
        #     ###################For NEEM#######################################
        #     a_1 = 3.5E-1
        #     a_2 = 1.359319E-2
        #     a_3 = -1.569421E-2
        #     a_4 = -4.3E-1
        #     a_5 = 4.332293E-1
        #     a_6 = 7.976252E-3
        #     a_7 = -3.536121E-5
        #     a_8 = 8.82746379E-1
        #     a_9 = 3.7853621E-2
        #     a_10 = -5.198599E-3
        #     z_1 = self.z[self.z < 16]
        #     z_2 = self.z[np.where((16 <= self.z) & (self.z < 110))]
        #     z_3 = self.z[self.z >= 110]
        #     rho_1 = a_1 + a_2 * z_1 + a_3 * np.exp(a_4 * (16 - z_1))
        #     rho_2 = a_5 + a_6 * z_2 + a_7 * z_2 * z_2
        #     rho_3 = a_8 + a_9 * (1 - np.exp(a_10 * (z_3 - 110)))
        #     rho = np.concatenate((rho_1, rho_2, rho_3))
        #     # np.savetxt(C.ROOT+"NEEM_EU\\density_smoothed.txt", np.hstack((self.z.reshape(self.z.shape[0],1), rho.reshape(self.z.shape[0],1))))
        #     return rho
        # else:
        #     df = pd.read_excel(self.S.rho_Profile)

    
        



    ##########Mitchell et al., 2015#########
    def _s(self):
        return 1 - self.rho / self.S.rho_ice

    def _s_cl(self):
        # z_before_COD = self.z[np.where(self.rho < self.rho_COD)]
        # z_after_COD = self.z[np.where(self.rho >= self.rho_COD)]

        #################Schwander et al., 1998############################
        # rho_before_COD = self.rho[np.where(self.rho < self.rho_COD)]
        # # rho_after_COD = z[np.where(rho >= self.rho_COD)]

        # s_before_COD = self.s[np.where(self.rho < self.rho_COD)]
        # s_after_COD = self.s[np.where(self.rho >= self.rho_COD)]

        # s_cl_before_COD = s_before_COD * np.exp(75 / self.rho_COD * (rho_before_COD - self.rho_COD))
        # s_cl_after_COD = s_after_COD

        # return np.concatenate((s_cl_before_COD, s_cl_after_COD))

        #################Goujon et al., 2003###############################
        s_co_bar = 1 - self.rho_COD_bar / self.S.rho_ice
        s_cl = 0.37 * self.s * np.power(self.s / s_co_bar, -7.6)
        s_cl[s_cl > self.s] = self.s[s_cl > self.s]

        return s_cl
        
    

    def _s_op(self):
        return self.s - self.s_cl
    
    def _s_op_star(self):
        return self.s_op * np.exp(C.M_air * C.g * self.z / C.R / self.S.T)
        # return self.s_op * self.rho / self.rho_LID
    
    def porosity(self):
        return self.s, self.s_cl, self.s_op, self.rho
    
    def _w_ice(self):
        #########Buizert et al., 2011###########
        # return self.S.A_ieq * self.S.rho_ice / self.rho
        #########Goujon et al., 2003###########
        zeta = self.z / self.S.H
        m = 10
        return self.S.rho_ice / self.rho * (self.S.A_ieq - self.S.A_ieq * ((m + 2) / (m + 1) * zeta) * (1 - (zeta ** (m + 1)) / (m + 2)))

    def _p_cl(self):
        p_cl = list()

        d_s_cl_d_z = np.gradient(self.s_cl, self.z)

        # d_s_cl_d_z = np.diff(self.s_cl) / self.Delta_z
        # d_s_cl_d_z = np.insert(d_s_cl_d_z, 0, 0)
        exp = np.exp(C.M_air * C.g * self.z / C.R / self.S.T)
        for i in range(0, self.COD_idx + 1):
            zeta = self.s[:i+1] / self.s[i] / (1 + np.log(self.w_ice[i] / self.w_ice[:i+1]))
            p_cl.append(np.trapz(d_s_cl_d_z[:i+1] * exp[:i+1] * zeta, self.z[:i+1]))
            p_cl[-1] /= np.trapz(d_s_cl_d_z[:i+1], self.z[:i+1])
            # p_cl[-1] /= self.s_cl[i]
            
        p_cl[0] = 1
        p_cl_z_COD = p_cl[self.COD_idx]

        for i in range(self.COD_idx + 1, self.N):
            zeta = self.s[self.COD_idx] / self.s[i] / (1 + np.log(self.w_ice[i] / self.w_ice[self.COD_idx]))
            p_cl.append(p_cl_z_COD * zeta)

        p_cl = np.array(p_cl)
        p_cl[p_cl < 1] = 1

        return p_cl

    def _w_air(self):
        # flux_COD = self.s_cl[self.COD_idx] * self.p_cl[self.COD_idx] / self.rho_COD
        # w_air_before_COD = self.S.A_ieq * self.S.rho_ice / (self.s_op_star[:self.COD_idx] + 1E-10) * (flux_COD + 1E-10 - self.s_cl[:self.COD_idx] * self.p_cl[:self.COD_idx] / self.rho[:self.COD_idx])
        
        flux_COD = self.w_ice[self.COD_idx+1] * self.s_cl[self.COD_idx+1] * self.p_cl[self.COD_idx+1]
        # w_air_before_COD = 1 / (self.s_op_star[:self.COD_idx] + 1E-10) * (flux_COD + 1E-10 - self.w_ice[:self.COD_idx] * self.s_cl[:self.COD_idx] * self.p_cl[:self.COD_idx])
       
        w_air = 1 / (self.s_op_star + 1E-10) * (flux_COD + 1E-10 - self.w_ice * self.s_cl * self.p_cl)
        close_idx = np.argmin(np.abs(w_air - self.w_ice))
        w_air[close_idx:] = self.w_ice[close_idx:]

        # w_air_after_COD = self.w_ice[self.COD_idx:]
        return w_air
    
    def velocity(self):
        return self.w_air, self.w_ice, self.p_cl
    

    ###########OSU Model##################
    def _tau_inv_DZ(self):
        inv_tort = np.loadtxt(C.ROOT+"icecores\\"+self.S.name+"\\tortuosity_DZ.txt")
        inv_tort = np.interp(self.z, inv_tort[:, 0], inv_tort[:, 1])
        inv_tort[inv_tort > 1] = 1
        return inv_tort

    def _tau_inv_LIZ(self):
        D_m = np.loadtxt(C.ROOT+"icecores\\"+self.S.name+"\\tortuosity_LIZ.txt")
        D_m = np.interp(self.z, D_m[:, 0], D_m[:, 1])
        return D_m
    
    def tortuosity(self):
        return self.tau_inv_DZ, self.tau_inv_LIZ

    def _D_X(self):
        
        # if self.S.name :   #From Tortuosity data
        # if self.S.name == "CPSW":
        #     g1 = -0.209
        #     g2 = 1.515
        #     g3 = 0.53
        #     g4 = 3.17E-10
        #     g5 = 1.82
        #     s_op = self.s_op#[:self.LID_idx + 1]
        #     # s_op = self.s_op_star[:self.LID_idx + 1]
        #     D_X_before_LID = self.D_X_0 * (g1 + g2 * s_op + g3 * s_op * s_op)
        #     self.LID_idx = np.min(np.where(D_X_before_LID < 0))
        #     D_X_before_LID = D_X_before_LID[:self.LID_idx]
        #     # D_CH4_before_LID[D_CH4_before_LID < 0] = 0

        #     self.rho_LID = self.rho[self.LID_idx]
        #     self.z_LID = self.z[self.LID_idx]

        #     D_X_after_LID = g4 + (D_X_before_LID[-1] - g4) * np.exp(-g5 * (self.z[self.LID_idx:] - self.z_LID))
        #     return np.concatenate((D_X_before_LID, D_X_after_LID))
        # else:
        # D_X = self.s_op * self.D_X_0 * inv_tort   #이거 아닌가
        D_X = self.D_X_0 * self.tau_inv_DZ
        
        self.rho_LID = self.rho_COD - 0.04 #0.014                             #Blunier et al., 2000
        self.LID_idx = np.argmin(np.abs(self.rho - self.rho_LID))
        self.z_LID = self.z[self.LID_idx]

        return D_X
    
    def _D_eddy(self):
        H = 4.5
        if self.S.name == "NEEM":   #From LIZ data
            D_eddy_0 = 2.30453E-5
            # D_eddy_0 = 1.6E-5
            D_eddy = D_eddy_0 * np.exp(-self.z / H)
            D_eddy[self.z >= 55] = 0
            D_eddy = np.maximum(D_eddy, self.tau_inv_LIZ)
            return D_eddy
        else:
            g6 = 3.17E-9
            g7 = 0.11
            #########################SIO Model#####################################
            D_eddy_before_LID = 1.6E-5 * np.exp(-self.z[:self.LID_idx + 1] / H)
            #########################OSU Model#####################################
            D_eddy_before_COD = g6 * np.exp(g7 * (self.z[self.LID_idx + 1:self.COD_idx + 1] - self.z_LID))
            D_eddy_after_COD = np.zeros(self.N - self.COD_idx - 1)
            return np.concatenate((D_eddy_before_LID, D_eddy_before_COD, D_eddy_after_COD))
    
    def diffusion(self):
        return self.D_X, self.D_eddy, self.D_X + self.D_eddy, self.s_op_star
    
    ##############CIC Model######################
    def _x_air(self):
        return self.s_cl[self.COD_idx] * self.p_cl[self.COD_idx] * self.S.p_0 / C.P0 * C.C_to_K / self.S.T / self.rho[self.COD_idx]

    def _C_i_n(self):
        Delta_M = self.M_X - C.M_air

        alpha = self.D_X + self.D_eddy
        _beta = 1 / self.s_op_star * np.gradient(self.s_op_star * (self.D_X + self.D_eddy), self.z)
        _beta[np.isnan(_beta)] = 0
        _beta[_beta == np.inf] = 0
        _beta[_beta == -np.inf] = 0
        beta = self.D_X * Delta_M * C.g / C.R / self.S.T - self.w_air / C.year_to_sec + _beta
        _gamma = 1 / self.s_op_star * np.gradient(self.s_op_star * self.D_X, self.z)
        _gamma[np.isnan(_gamma)] = 0
        _gamma[_gamma == np.inf] = 0
        _gamma[_gamma == -np.inf] = 0
        gamma = - Delta_M * C.g / C.R / self.S.T * _gamma - self.lambda_X

        alpha_i_star = alpha * self.Delta_t / 2 / self.Delta_z / self.Delta_z
        beta_i_star = beta * self.Delta_t / 4 / self.Delta_z
        gamma_i_star = gamma * self.Delta_t / 2


        A = np.zeros((self.M + 1, self.M + 1))
        for i in range(self.M + 1):
            for j in range(i - 1, i + 2):
                if j == -1 or j == self.M + 1:
                    continue
                if j == i - 1:
                    A[i,j] = -(alpha_i_star[i] - beta_i_star[i])
                elif j == i:
                    A[i,j] = 1 + 2 * alpha_i_star[i] - gamma_i_star[i]
                elif j == i + 1:
                    A[i,j] = -(alpha_i_star[i] + beta_i_star[i])
        A[0,0] = 1
        A[0,1] = 0
        A[self.M,self.M-1] = -1
        A[self.M,self.M] = 1


        A_inv = np.linalg.inv(A)

        B = np.zeros((self.M + 1, self.M + 1))
        for i in range(self.M + 1):
            for j in range(i - 1, i + 2):
                if j == -1 or j == self.M + 1:
                    continue
                if j == i - 1:
                    B[i,j] = alpha_i_star[i] - beta_i_star[i]
                elif j == i:
                    B[i,j] = 1 - 2 * alpha_i_star[i] + gamma_i_star[i]
                elif j == i + 1:
                    B[i,j] = alpha_i_star[i] + beta_i_star[i]

        C_atm = 0.0 * np.ones(int(self.T / C.dt) + 1)
        C_atm[int(0.2 / C.dt) : int(0.4 / C.dt)] = 1 / (0.4 - 0.2)
        # C_atm[2] = C_atm[3] = 1
        C_i_n = C_atm[0] * np.ones((self.M + 1, 1))

        for t in tqdm(range(0, int(self.T / C.dt))):

            C_n = C_i_n[:,-1]
            C_n = C_n[:,np.newaxis]

            # print('t:', t)
            BC_n = B @ C_n
            BC_n[0] = C_atm[t + 1]
            BC_n[self.M] = 0

            C_n = A_inv @ BC_n
            C_n[C_n < 0] = 0
            C_i_n = np.hstack((C_i_n, C_n))

        
        C_i_n_test = C.dt * np.sum(C_i_n, axis = 1)
        print(np.max(C_i_n_test), np.argmax(C_i_n_test), np.min(C_i_n_test), np.argmin(C_i_n_test))

        T, Z = np.meshgrid(range(0, int(self.T / C.dt) + 1), self.z)
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1,projection = "3d")
        ax.plot_surface(Z,T,C_i_n)
        plt.show()
        return C_i_n
    
    def gas_age_distribution(self):
        return self.C_i_n[self.LID_idx, :], self.C_i_n[self.COD_idx, :], self.C_i_n
    
    def _Delta(self, depth):
        G = self.C_i_n[np.argmin(np.abs(self.z - depth)), :]
        t = np.arange(0, self.T + C.dt, C.dt)
        Gamma = np.trapz(t * G, t)
        Delta = np.sqrt(0.5 * np.trapz((t - Gamma) * (t - Gamma) * G, t))
        return Gamma, Delta

    def _FWHM(self, depth):
        G = self.C_i_n[np.argmin(np.abs(self.z - depth)), :]
        median = np.argmax(G)
        half_max = G[median] / 2
        mx = np.max(np.where(G - half_max > 0))
        mn = np.min(np.where(G - half_max > 0))
        FWHM = C.dt * (mx - mn + 1.5)
        return (mx + mn) / 2 * C.dt, FWHM
    
    def printAgeIndicators(self, depth):
        Gamma, Delta = self._Delta(depth)
        median, FWHM = self._FWHM(depth)
        print("Depth : ", depth, "Gamma : ", Gamma, "Median : ", median, "FWHM : ", FWHM, "Delta : ", Delta)


