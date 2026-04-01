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
        self.Nt = int((C.Time + (C.dt / 100)) / C.dt)

        self.t = np.arange(0.0, C.Time + C.dt / 2, C.dt)

        self.S = site
        self.G = gas


        # self.D_CO2_0 = 1.39E-5 * (self.S.T/C.C_to_K) ** 1.75 * (C.P0 / self.S.p_0)
        self.D_CO2_0 = 5.75E-10 * self.S.T ** 1.81 * (C.P0 / self.S.p_0)
        # self.D_CO2_0 = 1.638946715E-05  # Buizert's MATLAB code
        self.D_X_0 = gas.gamma_X * self.D_CO2_0

        self.Delta_z = C.dz
        self.Delta_t = C.dt * C.year_to_sec #년을 초로

        #######Herron and Langway, 1980#########
        self.rho = self._rho()

        ##############Mitchell et al., 2015##############
        # self.rho_COD = 1 / (1 - 1 / 75) / (1 / C.kg_to_g / self.S.rho_ice + 7.02E-7 * self.S.T - 4.5E-5) / C.kg_to_g 
        # self.rho_COD = 75 / 74 / (1 / self.S.rho_ice + self.S.T * 6.95E-4 - 4.3E-2)
        self.rho_COD_bar = 1 / (1 / self.S.rho_ice + 6.95E-4 * self.S.T - 4.3E-2)    #Martinerie et al., 1994, Mitchell et al., 2015
        
        # self.COD_idx = np.argmin(np.abs(self.rho - self.rho_COD))
        # self.rho_COD = self.rho[self.COD_idx]
        # self.z_COD = self.z[self.COD_idx]


        # self.rho_LID = self.rho_COD - 0.014                             #Blunier et al., 2000
        # self.LID_idx = np.argmin(np.abs(self.rho - self.rho_LID))


        self.s = self._s()
        self.s_cl = self._s_cl()
        self.s_op = self._s_op()
        self.s_op_star = self._s_op_star()


        self.COD_idx = np.argmax(self.s_cl)
        self.rho_COD = self.rho[self.COD_idx]
        self.z_COD = self.z[self.COD_idx]

        self.M = np.argmin(np.abs(self.z - (self.z_COD + 10)))
        
        self.C_shape = (self.M + 1, self.Nt + 1)
        # self.M = np.argmin(np.abs(self.z - self.S.Z))

        ################OSU Model########################
        self.tau_inv_DZ = self._tau_inv_DZ()
        self.tau_inv_LIZ = self._tau_inv_LIZ()

        self.D_X = self._D_X()
        self.D_eddy = self._D_eddy()
        print("rho_LID : ", self.rho_LID, " rho_COD : ", self.rho_COD)
        print("LID : ", self.z_LID, "m, COD : ", self.z_COD, "m")

        ##############Buizert et al., 2011##############
        self.w_ice = self._w_ice()
        self.p_cl = self._p_cl()
        self.w_air = self._w_air()


        self.phi_op = self._phi_op()
        self.phi_cl = self._phi_cl()

        ################CIC Model#########################
        self.x_air = self._x_air()
        print("x_air : ", self.x_air * 1E3, "ml/kg")
        self.C_op = self._C_op()
        self.eta = self._eta()
        self.C_cl, self.C_total = self._C_cl()

        # self.Gamma, self.Delta = self._Delta()
        # self.median, self.FWHM = self._FWHM()

    def _rho(self):
        if self.S.use_HL:
            #######Herron and Langway, 1980#########
            k_0 = 11 * np.exp(-10160 / C.R / self.S.T)
            k_1 = 575 * np.exp(-21400 / C.R / self.S.T)
            z_crit = 1 / k_0 / self.S.rho_ice * (np.log(0.55 / (self.S.rho_ice - 0.55)) - np.log(self.S.rho_0 / (self.S.rho_ice - self.S.rho_0)))
            print("z_0.55 : ", z_crit)
            Z_0 = np.exp(self.S.rho_ice * k_0 * self.z[self.z < z_crit] + np.log(self.S.rho_0 / (self.S.rho_ice - self.S.rho_0)))
            rho_shallow = self.S.rho_ice * Z_0 / (1 + Z_0)

            Z_1 = np.exp(self.S.rho_ice * k_1 * (self.z[self.z >= z_crit] - z_crit) / self.S.A_weq ** 0.5 + np.log(0.55 / (self.S.rho_ice - 0.55)))
            rho_deep = self.S.rho_ice * Z_1 / (1 + Z_1)

            # z_crit_idx = np.argmin(np.abs(rho_deep - rho_shallow))
            return np.concatenate((rho_shallow, rho_deep))
            # return np.concatenate((rho_shallow[:z_crit_idx], rho_deep[z_crit_idx:]))
        else:
            rho = np.loadtxt(C.ROOT+"icecores\\"+self.S.name+"\\density_smoothed.txt")
            rho = np.interp(self.z, rho[:, 0], rho[:, 1])
            return rho

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
        s_op = self.s - self.s_cl
        s_op[s_op < 1E-9] = 1E-9
        return s_op
    
    def _s_op_star(self):
        return self.s_op * np.exp(C.M_air * C.g * self.z / C.R / self.S.T)
        # return self.s_op * self.rho / self.rho_LID
    
    def porosity(self):
        return self.s, self.s_cl, self.s_op, self.rho
    
    
    def _w_ice(self):
        #########Buizert et al., 2011###########
        return self.S.A_ieq * self.S.rho_ice / self.rho
        #########Goujon et al., 2003###########
        # zeta = self.z / self.S.H
        # m = 10
        # return self.S.rho_ice / self.rho * (self.S.A_ieq - self.S.A_ieq * ((m + 2) / (m + 1) * zeta) * (1 - (zeta ** (m + 1)) / (m + 2)))

    def _p_cl(self):

        p_cl = np.zeros(self.N)
        # dz = self.z[1] - self.z[0] # dz (단일 값) 추출
        
        strain = np.gradient(np.log(self.w_ice), self.z)

        # 1. 미분 및 상수 준비
        # MATLAB의 dscl이 s_cl의 미분값이라면 np.gradient를 유지하되, 
        # 만약 diff를 썼다면 np.diff 후 zero-padding이 필요할 수 있습니다.
        # dscl = np.gradient(self.s_cl, self.z)
        dscl = np.concatenate([[0], np.diff(self.s_cl) / C.dz])
        exp_term = np.exp(C.M_air * C.g * self.z / C.R / self.S.T)

        # 2. COD 지점까지의 루프 (MATLAB의 teller1 = 1:teller_co)
        for i in range(self.COD_idx + 1):
            integral_num = []
            integral_den = []
            
            for j in range(i + 1):
                # MATLAB: 1 + Trapezoid(strain(j:i), dz)
                # 주의: MATLAB의 j:i는 i번째 인덱스를 포함하므로 i+1까지 슬라이싱
                zeta_denom = 1 + np.trapz(strain[j:i+1], dx=C.dz)
                
                # MATLAB: dscl(j) * C(j) * (s(j)/s(i)) / zeta_denom
                val_num = dscl[j] * exp_term[j] * (self.s[j] / self.s[i]) / zeta_denom
                integral_num.append(val_num)
                integral_den.append(dscl[j])

            # MATLAB: (dz * sum(integral)) / (dz * sum(integral2))
            if np.sum(integral_den) != 0:
                p_cl[i] = (C.dz * np.sum(integral_num)) / (C.dz * np.sum(integral_den))
            else:
                p_cl[i] = 1

        # 3. COD 이후 지점 (MATLAB의 teller1 = (teller_co+1):length(z))
        p_cl_z_COD = p_cl[self.COD_idx]
        for i in range(self.COD_idx + 1, self.N):
            # MATLAB: bubble_pres(teller_co) * (s(co)/s(i)) / (v_ice(i)/v_ice(co))
            # v_ice는 self.w_ice와 대응된다고 가정
            ratio_s = self.s[self.COD_idx] / self.s[i]
            ratio_v = self.w_ice[i] / self.w_ice[self.COD_idx]
            p_cl[i] = p_cl_z_COD * ratio_s / ratio_v

        # 예외 처리 및 초기값
        p_cl[0] = 1
        p_cl[p_cl < 1] = 1

        return p_cl

    def _w_air(self):
        # flux_COD = self.s_cl[self.COD_idx] * self.p_cl[self.COD_idx] / self.rho_COD
        # w_air_before_COD = self.S.A_ieq * self.S.rho_ice / (self.s_op_star[:self.COD_idx] + 1E-10) * (flux_COD + 1E-10 - self.s_cl[:self.COD_idx] * self.p_cl[:self.COD_idx] / self.rho[:self.COD_idx])
        
        flux_COD = self.w_ice[self.COD_idx+1] * self.s_cl[self.COD_idx+1] * self.p_cl[self.COD_idx+1]
        # w_air_before_COD = 1 / (self.s_op_star[:self.COD_idx] + 1E-10) * (flux_COD + 1E-10 - self.w_ice[:self.COD_idx] * self.s_cl[:self.COD_idx] * self.p_cl[:self.COD_idx])
        # flux = self.w_ice * self.p_cl * self.s_cl
        # plt.scatter(self.z, self.w_ice, marker='.', c='r', s=1)
        # plt.scatter(self.z, self.p_cl, marker='.', c='b', s=1)
        # plt.scatter(self.z, self.s_cl, marker='.', c='g', s=1)
        # plt.scatter(self.z, flux, marker='.', s=1)
        # plt.show()
        w_air = (flux_COD + 1E-10 - self.w_ice * self.p_cl * self.s_cl) / (self.s_op_star + 1E-10)
        w_air = np.minimum(self.w_ice, w_air)
        w_air[self.COD_idx:] = self.w_ice[self.COD_idx:]
        # w_air = 1 / (self.s_op_star + 1E-10) * (flux_COD + 1E-10 - self.w_ice * self.s_cl * self.p_cl)
        # close_idx = np.argmin(np.abs(w_air - self.w_ice))
        # w_air[close_idx:] = self.w_ice[close_idx:]

        # w_air_after_COD = self.w_ice[self.COD_idx:]
        return w_air
    
    def velocity(self):
        return self.w_air, self.w_ice, self.p_cl
    
    def _phi_op(self):
        return self.s_op_star * self.w_air
    
    def _phi_cl(self):
        return self.s_cl * self.p_cl * self.w_ice

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
        # D_X = self.s_op_star * self.D_X_0 * self.tau_inv_DZ   #이거 아닌가
        D_X = self.D_X_0 * self.tau_inv_DZ
        
        self.rho_LID = self.rho_COD - 0.01 #0.014                             #Blunier et al., 2000
        self.LID_idx = np.argmin(np.abs(self.rho - self.rho_LID))
        self.z_LID = self.z[self.LID_idx]

        return D_X
    
    def _D_eddy(self):
        H = 4.5
        if self.S.name == "NEEM_EU":
            D_eddy_0 = 2.30453E-5
        elif self.S.name == "NEEM_US":
            D_eddy_0 = 2.426405E-5
        else:
            D_eddy_0 = 2.30453E-5
        # D_eddy_0 = 1.6E-5
        D_eddy = D_eddy_0 * np.exp(-self.z / H)
        D_eddy[self.z >= 55] = 0
        D_eddy = np.maximum(D_eddy, self.tau_inv_LIZ)
        return D_eddy
        # if self.S.name in ["NEEM_EU", "NEEM_US"]:   #From LIZ data
        #     D_eddy_0 = 2.30453E-5
        #     # D_eddy_0 = 1.6E-5
        #     D_eddy = D_eddy_0 * np.exp(-self.z / H)
        #     D_eddy[self.z >= 55] = 0
        #     D_eddy = np.maximum(D_eddy, self.tau_inv_LIZ)
        #     return D_eddy
        # else:
        #     g6 = 3.17E-9
        #     g7 = 0.11
        #     #########################SIO Model#####################################
        #     D_eddy_before_LID = 1.6E-5 * np.exp(-self.z[:self.LID_idx + 1] / H)
        #     #########################OSU Model#####################################
        #     D_eddy_before_COD = g6 * np.exp(g7 * (self.z[self.LID_idx + 1:self.COD_idx + 1] - self.z_LID))
        #     D_eddy_after_COD = np.zeros(self.N - self.COD_idx - 1)
        #     return np.concatenate((D_eddy_before_LID, D_eddy_before_COD, D_eddy_after_COD))
    
    def diffusion(self):
        return self.D_X, self.D_eddy, self.D_X + self.D_eddy, self.s_op_star
    
    ##############CIC Model######################
    def _x_air(self):
        return self.s_cl[self.COD_idx] * self.p_cl[self.COD_idx] * self.S.p_0 / C.P0 * C.C_to_K / self.S.T / self.rho[self.COD_idx]

    def _C_op(self):
        Delta_M = self.G.M_X - C.M_air

        alpha = self.D_X + self.D_eddy
        _beta = 1 / self.s_op_star * np.gradient(self.s_op_star * (self.D_X + self.D_eddy), self.z)
        _beta[self.COD_idx] = _beta[self.COD_idx + 1]
        beta = -self.D_X * (Delta_M * C.g) / (C.R * self.S.T) + self.D_eddy * (C.M_air * C.g) / (C.R * self.S.T) - (self.w_air) / (C.year_to_sec) + _beta
        _gamma = 1 / self.s_op_star * np.gradient(self.s_op_star * self.D_X, self.z)
        _gamma[self.COD_idx] = _gamma[self.COD_idx + 1]
        gamma = -Delta_M * C.g / C.R / self.S.T * _gamma - self.G.lambda_X

        alpha_i_star = alpha * self.Delta_t / (2 * self.Delta_z * self.Delta_z)
        beta_i_star = beta * self.Delta_t / (4 * self.Delta_z)
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


        A_inv = np.linalg.pinv(A)

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

        # C_atm[2] = C_atm[3] = 1
        # C_i_n = C_atm[0] * np.ones((self.M + 1, 1))

        C_atm = np.zeros(self.Nt + 1)
        C_atm[(0.2 - 1E-9 <= self.t) & (self.t < 0.4 - 1E-9)] = 1 / (0.4 - 0.2)
        C_op = np.zeros(self.C_shape)
        C_op[:, 0] = C_atm[0]

        for t in tqdm(range(self.Nt)):
            C_n_current = C_op[:, t:t+1]
            BC_n = B @ C_n_current
            BC_n[0] = C_atm[t + 1]
            BC_n[self.M] = 0
            
            C_next = A_inv @ BC_n
            C_next[C_next < 0] = 0
            C_op[:, t+1] = C_next.flatten()
        
        
        C_i_n_test = C.dt * np.sum(C_op, axis = 1)
        print(np.max(C_i_n_test), np.argmax(C_i_n_test), np.min(C_i_n_test), np.argmin(C_i_n_test))
        C_op[self.COD_idx:, :] = 0.0
        # T, Z = np.meshgrid(range(0, int(self.T / C.dt) + 1), self.z)
        # fig = plt.figure(figsize = (6,6))
        # ax = fig.add_subplot(1,1,1,projection = "3d")
        # ax.plot_surface(Z,T,C_i_n, cmap='coolwarm')
        # plt.show()
        return C_op
    
    def _eta(self):
        eta = np.zeros(self.M + 1)
        for i in range(0, self.M + 1):
            eta[i] = np.trapz(1 / self.w_ice[:i], self.z[:i])
        return eta

    def _C_cl(self):
        trapping_t = np.gradient(self.phi_cl, self.z) / self.w_ice

        C_cl = np.zeros(self.C_shape)
        C_total = np.zeros(self.C_shape)
        M_position = np.zeros(self.C_shape)
        M_C = np.ones(self.C_shape)
        M_position[:, -1] = self.z[:self.M + 1].copy()
        for i in range(1, self.Nt + 1):
            M_position[:, self.Nt - i] = np.interp(self.eta - i * C.dt, self.eta, self.z[:self.M + 1])
        M_trapping = np.interp(M_position, self.z, trapping_t)
        M_C[M_position < 1E-9] = 0

        full_air = self.s_op_star + self.s_cl * self.p_cl

        for i in tqdm(range(0, self.Nt + 1)):
            M_C_open = np.zeros(self.C_shape)
            M_C_open[:, self.Nt] = self.C_op[:, self.Nt - i]
            for j in range(1, self.Nt - i + 1):
                M_C_open[:, self.Nt - j] = np.interp(M_position[:, self.Nt - j], self.z[:self.M + 1], self.C_op[:, self.Nt - j - i])
            C_cl[:, self.Nt - i] = np.sum(M_C_open * M_trapping, axis=1) / np.sum(M_C * M_trapping, axis=1)
            C_total[:, self.Nt - i] = (C_cl[:, self.Nt - i] * self.s_cl[:self.M + 1] * self.p_cl[:self.M + 1] + self.C_op[:, self.Nt - i] * self.s_op_star[:self.M + 1]) / full_air[:self.M + 1]
        
        return C_cl, C_total


    # def gas_age_distribution(self):
    #     return self.C_op[self.LID_idx, :], self.C_op[self.COD_idx, :], self.C_op
    
    def _Delta(self, depth):
        def delta(C_):
            G_total = C_[np.argmin(np.abs(self.z - depth)), :]
            if np.max(G_total) < 1E-9:
                return 0, 0 
            Gamma = np.trapz(self.t * G_total, self.t)
            Delta = np.sqrt(0.5 * np.trapz((self.t - Gamma) * (self.t - Gamma) * G_total, self.t))
            return Gamma, Delta
        gamma_total, delta_total = delta(self.C_total)
        gamma_cl, delta_cl = delta(self.C_cl)
        gamma_op, delta_op = delta(self.C_op)
        gammas = [gamma_total, gamma_cl, gamma_op]
        deltas = [delta_total, delta_cl, delta_op]
        return gammas, deltas

    def _FWHM(self, depth):
        def fwhm(C_):
            G = C_[np.argmin(np.abs(self.z - depth)), :]
            median = np.argmax(G)
            half_max = G[median] / 2
            if half_max <= 1E-9:
                return 0, 0
            mx = np.max(np.where(G - half_max > 0))
            mn = np.min(np.where(G - half_max > 0))
            FWHM = C.dt * (mx - mn + 1.5)
            return (mx + mn) / 2 * C.dt, FWHM
        median_total, fwhm_total = fwhm(self.C_total)
        median_cl, fwhm_cl = fwhm(self.C_cl)
        median_op, fwhm_op = fwhm(self.C_op)
        medians = [median_total, median_cl, median_op]
        fwhms = [fwhm_total, fwhm_cl, fwhm_op]
        return medians, fwhms
    
    def plotGAD(self, depth):
        Gamma, Delta = self._Delta(depth)
        median, FWHM = self._FWHM(depth)
        print("[Total] Depth : ", round(depth,2) , "Gamma : ", round(Gamma[0], 2), "Median : ", round(median[0], 2), "FWHM : ", round(FWHM[0], 2), "Delta : ", round(Delta[0], 2))
        print("[Opened] Depth : ", round(depth,2) , "Gamma : ", round(Gamma[1], 2), "Median : ", round(median[1], 2), "FWHM : ", round(FWHM[1], 2), "Delta : ", round(Delta[1], 2))
        print("[Closed] Depth : ", round(depth,2) , "Gamma : ", round(Gamma[2], 2), "Median : ", round(median[2], 2), "FWHM : ", round(FWHM[2], 2), "Delta : ", round(Delta[2], 2))
        idx = np.argmin(np.abs(self.z - depth))
        x = self.S.sample_year - self.t
        ice_age = self.S.sample_year - self.eta[idx]
        total_age = self.S.sample_year - Gamma[0]
        ls = ":"
        plt.axvline(ice_age, c="dodgerblue", linestyle="dashed", label="Ice Year")
        plt.axvline(total_age, c="g", linestyle=ls)
        plt.text(ice_age, 0, str(round(ice_age, 2)), ha="center", va="bottom", c="dodgerblue")
        plt.text(total_age, 0, str(round(total_age, 2)), ha="center", va="bottom", c="g")
        plt.plot(x, self.C_total[idx, :], c='g', label="C_total")

        if idx < self.COD_idx:
            closed_age = self.S.sample_year - Gamma[1]
            opened_age = self.S.sample_year - Gamma[2]

            plt.axvline(closed_age, c="b", linestyle=ls)
            plt.axvline(opened_age, c="r", linestyle=ls)
            
            plt.text(closed_age, 0, str(round(closed_age, 2)), ha="center", va="bottom", c="b")
            plt.text(opened_age, 0, str(round(opened_age, 2)), ha="center", va="bottom", c="r")

            plt.plot(x, self.C_cl[idx, :], c="b", label="C_cl")
            plt.plot(x, self.C_op[idx, :], c="r", label="C_op")
        # plt.inver
        ymin, ymax = plt.ylim()
        plt.ylim(0, ymax)
        plt.grid(True)
        plt.legend()
        plt.title("Depth: "+str(depth)+" [m],\nDrilled Year: "+str(self.S.sample_year)+" [CE]")
        plt.show()

