import utils.constants as C

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

PLOT_INTERVAL = 1

class Profile:
    
    def __init__(self, site, gas):
        self.S = site
        self.G = gas

        self.z = np.arange(0.0, self.S.Z + C.dz / 2, C.dz)

        self.dz_R = np.diff(self.z)
        self.dz_R = np.append(self.dz_R, self.dz_R[-1])
        self.dz_L = np.delete(np.insert(self.dz_R, 0, self.dz_R[0]), -1)
        self.dz_S = (self.dz_L + self.dz_R) / 2
        self.Nz = self.z.shape[0]

        # self.t = np.arange(0.0, C.Time + C.dt / 2, C.dt)
        self.t = np.arange(self.S.sample_year - C.Time, self.S.sample_year + C.dt / 2, C.dt)
        # self.Nt = int((C.Time + (C.dt / 100)) / C.dt)
        self.Nt = self.t.shape[0]

        
        #######Goujon et al., 2003#########
        self.T_surf = self._init_T_surf()   #[1xNt]
        self.T_basal = self._init_T_basal() #[1xNt]
        self.T = self._init_T()             #[NzX1]


        self.rho_ice = self._init_rho_ice() #[Nzx1]

        self.A_ieq = self._init_A_ieq()     #[1xNt]
        self.A_weq = self._init_A_weq()     #[1x1], for H-L model


        self.C_atm = self._init_C_atm()
        self.p_atm = self._init_p_atm()     #[1xNt]


        #######Herron and Langway, 1980#########
        self.rho = self._init_rho() #[Nzx1], Mg/m3

        ##############Mitchell et al., 2015##############
        self.rho_COD_bar = self._init_rho_COD_bar() #[1xNt]



        self.s = self._init_s()
        self.s_cl = self._init_s_cl()

        self.COD_idx, self.rho_COD, self.z_COD = self._init_COD()

        self.M = np.argmin(np.abs(self.z - (self.z_COD + 10)))
        #######Buizert et al., 2016#########
        self.p_op, self.p_gas = self._init_p_op()       #[Nzx1]
        
        # self.plotBoundaryConditions()        
        self.D_X_0 = self._init_D_X_0()     #[Nzx1]


        self.s_op, self.s_op_safe, self.s_op_star = self._init_s_op()

        self.C_op, self.C_gas = self._init_C_op()

        self.iez = self._init_iez()
        self.Xi = self._init_Xi()
        self.rho_LID = self._init_rho_LID()
        self.z_LID = self._init_z_LID()

        self.C_shape = (self.Nz, self.Nt)
        # self.M = np.argmin(np.abs(self.z - self.S.Z))

        ################CIC Model######################## (From Buizert's code)
        # s와 관계있게 설정해야 될듯. 일단은 constant
        self.tau_inv_DZ = self._init_tau_inv_DZ()
        self.tau_inv_LIZ = self._init_tau_inv_LIZ()

        self.D_X = self._init_D_X()
        self.D_eddy = self._init_D_eddy()
        print("rho_LID : ", self.rho_LID, " rho_COD : ", self.rho_COD)
        print("LID : ", self.z_LID, "m, COD : ", self.z_COD, "m")

        ##############Buizert et al., 2011##############
        self.w_ice = self._init_w_ice()
        self.p_cl = self._init_p_cl()
        self.w_air = self._init_w_air()
        self.phi_op = self._init_phi_op()
        self.phi_cl = self._init_phi_cl()
        ################CIC Model#########################
        self.x_air = self._init_x_air()
        self.eta = self._init_eta()

        self.C_op, self.C_gas = self._init_C_op()
        self.C_cl, self.C_total = self._init_C_cl()

        # self.Gamma, self.Delta = self._Delta()
        # self.median, self.FWHM = self._FWHM()

    def run(self):

        plt.ion()
        self.plot_boundary_conditions()
        self.plot_state(title=f"Init (t = {self.t[0]:.1f} yr)")

        for i, t in tqdm(enumerate(self.t)):
            T_next = self._update_T(i)
            rho_next = self._update_rho(i)
            p_op_next, p_gas_next = self._update_p_op(i)

            rho_COD_bar_next = self._update_rho_COD_bar(i)
            s_next = self._update_s(i, rho_next)
            s_cl_next = self._update_s_cl(i, s_next, rho_COD_bar_next)
            COD_idx_next, rho_COD_next, z_COD_next = self._update_COD(rho_next, s_cl_next)

            C_op_next, C_gas_next = self._update_C_op(i, COD_idx_next)

            s_op_next, s_op_safe_next, s_op_star_next = self._update_s_op(T_next, p_gas_next, s_next, s_cl_next, COD_idx_next)

            iez_next = self._update_iez(i, rho_next)
            Xi_next = self._update_Xi(iez_next)
            w_ice_next = self._update_w_ice(i, rho_next, iez_next)

            p_cl_next = self._update_p_cl(i, T_next, rho_next, p_gas_next, w_ice_next, s_cl_next)

            C_cl_next, C_total_next = self._update_C_cl(i, C_gas_next, w_ice_next, s_cl_next, p_gas_next, p_cl_next, s_op_star_next)

            phi_cl_next = self._update_phi_cl(T_next, s_cl_next, p_cl_next, w_ice_next)
            w_air_next = self._update_w_air(T_next, Xi_next, w_ice_next, s_cl_next, p_cl_next, s_op_star_next, COD_idx_next)

            phi_op_next = self._update_phi_op(s_op_star_next, w_air_next)

            self.T = T_next
            self.rho = rho_next
            self.p_op, self.p_cl, self.p_gas = p_op_next, p_cl_next, p_gas_next
            self.C_op, self.C_gas = C_op_next, C_gas_next
            self.C_cl, self.C_total = C_cl_next, C_total_next
            self.s, self.s_cl, self.s_op, self.s_op_safe, self.s_op_star = s_next, s_cl_next, s_op_next, s_op_safe_next, s_op_star_next
            self.w_ice, self.w_air = w_ice_next, w_air_next
            self.phi_op, self.phi_cl = phi_op_next, phi_cl_next
            self.iez = iez_next
            self.Xi = Xi_next
            self.COD_idx, self.rho_COD_bar, self.rho_COD, self.z_COD = COD_idx_next, rho_COD_bar_next, rho_COD_next, z_COD_next

            if i % PLOT_INTERVAL == 0:
                self.plot_state(title=f"t = {self.t[i]:.2f} yr", t=self.t[i])

        plt.ioff()
        plt.show()

    def _thomas_solve(self, a, b, c, d):
        N = len(d)
        c_ = c.copy()
        d_ = d.copy()
        b_ = b.copy()

        for j in range(1, N):
            m = a[j] / b_[j - 1]
            b_[j] -= m * c_[j - 1]
            d_[j] -= m * d_[j - 1]

        x = np.empty(N)
        x[-1] = d_[-1] / b_[-1]
        for j in range(N - 2, -1, -1):
            x[j] = (d_[j] - c_[j] * x[j + 1]) / b_[j]

        return x
 
    def _newton_raphson_solve(self, t, G, G_grad, next_rho, previous_rho):
        """
        Newton-Raphson 1회 sweep (위→아래 순차).

        F(ρ)  = (ρ − ρ^n)/dt_sec + w(ρ)(ρ − ρ_{i-1})/dz_L − ρ_ice · G(D)
        F'(ρ) = 1/dt_sec + w_ice·ρ_ice·ρ_{i-1}/(ρ²·dz_L) − dG/dD
        """
        dt_sec = C.dt * C.year_to_sec
        rho_ice = self.rho_ice[t + 1]
        w = self.w_ice      # m/yr → m/s

        F_arr       = np.zeros(self.Nz)
        F_prime_arr = np.zeros(self.Nz)
        for j in range(1, len(next_rho)):
            rho_j = next_rho[j]
            rho_j_upper = next_rho[j - 1]

            # w_j = w[j] * rho_ice / rho_j

            F = (rho_j - previous_rho[j]) / dt_sec \
                + w[j] * (rho_ice / rho_j) * (rho_j - rho_j_upper) / self.dz_L[j] \
                - rho_ice * G[j]

            F_prime = 1 / dt_sec \
                + (w[j] * rho_ice * rho_j_upper) / (rho_j ** 2 * self.dz_L[j]) \
                - G_grad[j]
            

            F_arr[j]       = F
            F_prime_arr[j] = F_prime

            if np.abs(F_prime) < 1e-5:
                continue
            
            next_rho[j] = rho_j - F / F_prime

        debug = False
        if debug:
            fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
            axes[0].plot(F_arr, self.z, color='tab:blue')
            axes[0].set_xlabel("F")
            axes[0].set_ylabel("Depth [m]")
            axes[0].set_ylim(self.z[-1], self.z[0])
            axes[0].axhline(self.z_COD, color='k', linestyle='--', linewidth=0.8, label='COD')
            axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
            axes[0].grid(True, linestyle='--', alpha=0.5)
            axes[0].legend(fontsize=6)

            axes[1].plot(F_prime_arr, self.z, color='tab:orange')
            axes[1].set_xlabel("F'")
            axes[1].set_ylim(self.z[-1], self.z[0])
            axes[1].axhline(self.z_COD, color='k', linestyle='--', linewidth=0.8)
            axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
            axes[1].grid(True, linestyle='--', alpha=0.5)

            fig.suptitle(f't = {self.t[t]:.2f} yr', fontsize=12)
            plt.tight_layout()
            plt.show(block=True)

        return next_rho
    #Goujon et al., 2003
    def _update_T(self, t):
        """
        Goujon et al. (2003), Eq. (5) — 피른/빙상 내 열확산 모델
        Backward Euler (fully implicit) + Thomas algorithm

        ================================================================
        원본 PDE (논문 Eq. 5):
        ================================================================

            ρc ∂T/∂t = ∂/∂z(K ∂T/∂z) − ρcw ∂T/∂z

        여기서:
            T(z,t)  = 온도 [°C]
            ρ       = 피른 밀도 [kg/m³]
            c       = 비열 [J/(kg·K)]
            K       = 피른 열전도도 [W/(m·K)] = [kg·m/(s³·K)]
            w       = 수직 속도 [m/s]

        ================================================================
        열물성 (Table 1):
        ================================================================
            K_ice  = 2.22 (1 − 0.0067 T)                        [W/(m·K)]
            K_firn = K_ice (ρ/ρ_ice)^(2 − 0.5 ρ/ρ_ice)         [W/(m·K)]
            c_ice  = 152.5 + 7.122 T[K]                         [J/(kg·K)]
            (cρ)_firn = c_ice · ρ_firn                          [J/(m³·K)]

        ================================================================
        Conservative form 이산화 (dK/dz 항 자동 포함):
        ================================================================

        확산항을 conservative form으로 이산화:
            ∂/∂z(K ∂T/∂z) ≈ (K_{i+1/2}(T_{i+1}-T_i)/dz_R − K_{i-1/2}(T_i-T_{i-1})/dz_L) / dz_S

            K_{i+1/2} = (K_i + K_{i+1}) / 2
            K_{i-1/2} = (K_{i-1} + K_i) / 2

        이류항 (upwind):
            ρcw ∂T/∂z ≈ (ρc)_i w_i (T_i − T_{i-1}) / dz_L

        Backward Euler 정리:
            sub[i]  · T_{i-1}^{n+1} + diag[i] · T_i^{n+1} + sup[i] · T_{i+1}^{n+1} = rhs[i]

        ================================================================
        경계조건:
        ================================================================
            z=0 (표면)  : T = T_surf(t)    (Dirichlet)
            z=H (바닥)  : T = T_basal(t)   (Dirichlet)

        [ diag[0]  sup[0]                    ]   [ x[0] ]   [ rhs[0] ]
        [ sub[1]   diag[1]  sup[1]           ]   [ x[1] ]   [ rhs[1] ]
        [          sub[2]   diag[2]  sup[2]  ] × [ x[2] ] = [ rhs[2] ]
        [                   sub[3]   diag[3] ]   [ x[3] ]   [ rhs[3] ]
        """
        
        dt_sec = C.dt * C.year_to_sec

        previous_T = self.T.copy()
        previous_rho = self.rho.copy()
        rho_ice = self.rho_ice[t + 1]

        K_ice = 2.22 * (1 - 0.0067 * (previous_T - C.T0))   # Wm-1K-1
        K = (K_ice * (previous_rho / rho_ice) ** (2.0 - 0.5 * previous_rho / rho_ice)) # Wm-1K-1

        c_ice = (152.5 + 7.122 * previous_T) #Jkg-1K-1
        crho_firn = c_ice * previous_rho * (C.m_to_cm ** 3) / C.kg_to_g            #JK-1m-3

        K_half_temp = 0.5 * (K[:-1] + K[1:])       # K_{i+1/2}, [Nz-1]
        K_half_L = np.empty(self.Nz)             # K_{i-1/2}, [Nz]
        K_half_R = np.empty(self.Nz)        # [Nz]로 확장
        K_half_L[1:] = K_half_temp
        K_half_L[0]  = K[0]
        K_half_R[:-1] = K_half_temp
        K_half_R[-1]  = K[-1]

        w = self.w_ice

        # --- 무차원수 ---
        r_diff_L = K_half_L * dt_sec / (self.dz_L * self.dz_S * crho_firn)       # 확산 (왼쪽)
        r_diff_R = K_half_R * dt_sec / (self.dz_R * self.dz_S * crho_firn)  # 확산 (오른쪽)
        r_adv    = w * dt_sec / self.dz_L                               # 이류 (upwind)

        # --- 삼중대각 계수 ---
        sub  = -(r_diff_L + r_adv)                 # T_{i-1} 계수
        diag = 1.0 + r_diff_L + r_diff_R + r_adv   # T_i 계수
        sup  = -r_diff_R                            # T_{i+1} 계수

        rhs = previous_T.copy()                              # 우변 = T^n

        # --- 경계조건 적용 ---
        # z=0 : Dirichlet (표면 온도)
        diag[0] = 1.0
        sup[0]  = 0.0
        sub[0]  = 0.0
        rhs[0]  = self.T_surf[t + 1]

        # z=H : Dirichlet (바닥 온도)
        diag[-1] = 1.0
        sub[-1]  = 0.0
        sup[-1]  = 0.0
        rhs[-1]  = self.T_basal[t + 1]

        # --- Thomas 알고리즘 ---
        next_T = self._thomas_solve(sub, diag, sup, rhs)

        next_T[0] = self.T_surf[t + 1]
        next_T[-1] = self.T_basal[t + 1]

        return next_T

    def _update_rho(self, t):
        """ 
        Goujon et al. (2003), Appendix A — 피른 조밀화 모델
        Eulerian frame + Newton-Raphson implicit 풀이

        ================================================================
        원본 ODE (Lagrangian → Eulerian 변환, Eq. A15):
        ================================================================

            dD/dt = (1/ρ_ice) (∂ρ/∂t + w ∂ρ/∂z)

        여기서 D = ρ/ρ_ice (상대 밀도), dD/dt는 조밀화율.

        Eulerian 이산화:
            (ρ_i^{n+1} − ρ_i^n) / dt + w_i (ρ_i^{n+1} − ρ_{i-1}^{n+1}) / dz_L = ρ_ice · G(D_i)

            G(D) = dD/dt : 조밀화율 함수 (아래 3단계)

        Newton-Raphson:
            F(ρ)  = (ρ − ρ^n)/dt + w(ρ)(ρ − ρ_{i-1})/dz_L − ρ_ice · G(D) = 0
            F'(ρ) = 1/dt + w_ice·ρ_ice·ρ_{i-1} / (ρ²·dz_L) − ρ_ice · dG/dD / ρ_ice
            ρ^{k+1} = ρ^k − F/F'

        ================================================================
        조밀화 3단계 (Arnaud et al., 2000):
        ================================================================

        Stage 1 — Snow (D < D_0): grain boundary sliding
            dD/dt = γ (P/D²)(1 − 5D/3)                          (Eq. A1)

            D_0 = 0.00226·T_s[°C] + 0.647                       (Eq. A2)
            γ는 Stage 1→2 연속 조건으로 결정

        Stage 2 — Firn (0.6 ≤ D < ~0.9): power law creep
            dD/dt = 5.3 A (D²D_0)^{1/3} (a/π)^{1/2} (P*/3)^n   (Eq. A3)

            A = 7.89E-3 exp(−Q/RT)         [MPa⁻³ s⁻¹]         (Eq. A5)
            P* = 4πP / (aZD)                                     (Eq. A4)
            n = 3, Q = 60 kJ/mol
            Z(D), a(D), l', l'' : Arzt (1982) 기하학 모델       (Eq. A6–A9)

        Stage 3a — Bubbly ice, cylindrical (0.9 ≤ D < 0.95):
            dD/dt = 2A [D(1−D) / (1−(1−D)^{1/n})^n] (2P_eff/n)^n  (Eq. A10)

            P_eff = P + P_atm − P_b                               (Eq. A11)
            P_b   = P_c · D(1−D_c) / (D_c(1−D))                   (Eq. A12)

        Stage 3b — Bubbly ice, spherical (D ≥ 0.95):
            dD/dt = (9/4) A (1−D) P_eff                           (Eq. A13)

            A = 1.2E-3 exp(−Q/RT)           [MPa⁻¹ s⁻¹]         (Eq. A14)

        Stage 간 전환 구간:
            CubicHermiteSpline 보간으로 부드럽게 연결

        ================================================================
        경계조건:
        ================================================================
            z=0 : ρ = ρ_0 (표면 밀도, 고정)

        ================================================================
        풀이 순서:
        ================================================================
            1. 현재 밀도로 상재하중 P, 상대밀도 D, 조밀화율 G(D) 계산
            2. Newton-Raphson 3회 반복으로 ρ^{n+1} 수렴
               (위에서 아래로 순차 풀이 — upwind 이류 반영)
        """
        def D_dot(D, P, gamma, previous_T, D_0, T_s):
            """
            조밀화율 G(D) = dD/dt 계산.
            3단계 + Hermite 보간 전환 구간.
            """
            G = np.zeros(D.shape[0])
            n = 3
            Q = 6.0E4
            
            # --- Stage 2 계수 (전체 배열) ---
            c_Z = 15.5
            Z_0 = 110.2 * D_0**3 - 148.594 * D_0**2 + 87.6166 * D_0 - 17
            l_prime = (D / D_0) ** (1/3)
            Z = Z_0 + c_Z * (l_prime - 1)
            l_2prime = l_prime + (
                4 * Z_0 * (l_prime-1)**2 * (2*l_prime+1)
                + c_Z * (l_prime-1)**3 * (3*l_prime+1)
            ) / (
                12 * l_prime * (4*l_prime - 2*Z_0*(l_prime-1) - c_Z*(l_prime-1)**2)
            )
            a_contact = (np.pi / 3 / Z / l_prime**2) * (
                3 * (l_2prime**2 - 1) * Z_0
                + l_2prime**2 * c_Z * (2*l_2prime - 3)
                + c_Z
            )
            A_creep = 7.89E-15 * np.exp(-Q / (C.R * previous_T))
            P_star  = 4 * np.pi * P / a_contact / Z / D

            # --- Stage 1→2 전환 인덱스 ---
            Dms  = D_0 + 0.009
            ind1 = np.argmax(D >= Dms)

            # --- Stage 2 (ind1 이후) ---
            G[ind1:] = (5.3 * A_creep[ind1:]
                        * (D[ind1:]**2 * D_0)**(1/3)
                        * (a_contact[ind1:] / np.pi)**0.5
                        * (P_star[ind1:] / 3)**n)
            
            # --- gamma iteration (Stage 1) ---
            G[:ind1] = gamma * (P[:ind1] / D[:ind1]**2) \
                        * (1 - (5/3) * D[:ind1])

            gfrac = 0.03
            cc = 0
            while G[ind1-1] < G[ind1]:
                gamma *= (1 + gfrac)
                G[:ind1] = gamma * (P[:ind1] / D[:ind1]**2) \
                            * (1 - (5/3) * D[:ind1])
                cc += 1
                if cc > 10000:
                    print('Goujon gamma 상향 미수렴')
                    break

            counter = 0
            while G[ind1-1] >= G[ind1]:
                gamma /= (1 + gfrac/2.0)
                G[:ind1] = gamma * (P[:ind1] / D[:ind1]**2) \
                            * (1 - (5/3) * D[:ind1])
                counter += 1
                if counter > 10000:
                    print('Goujon gamma 하향 미수렴')
                    break

            # gamma 확정 후 Stage 1 전체 재계산
            G[:ind1] = gamma * (P[:ind1] / D[:ind1]**2) \
                        * (1 - (5/3) * D[:ind1])
            if ind1 > 0:
                n_trans = 5
                i_start = max(0, ind1 - n_trans)
                G[i_start:ind1+1] = np.linspace(G[i_start], G[ind1], ind1 + 1 - i_start)

            # --- Stage 3a (D > 0.9) ---
            P_atm = self.p_atm[t]
            V_c   = (6.95E-4 * T_s - 0.0043) / C.m_to_cm**3 * C.kg_to_g
            D_c   = 1 / (V_c * self.rho_ice[t] * C.Mg_to_kg + 1)
            P_b   = P_atm * (D * (1 - D_c) / D_c / (1 - D))
            P_eff = np.maximum(P + P_atm - P_b, 0.0)

            m3 = D > 0.9
            G[m3] = (2 * A_creep[m3]
                    * (D[m3]*(1-D[m3]) / (1-(1-D[m3])**(1/n))**n)
                    * (2 * P_eff[m3] / n)**n)

            # --- Stage 3b (D > 0.98) ---
            A_creep_3b = 1.2E-3 * np.exp(-Q / (C.R * previous_T))
            m4 = D > 0.98
            G[m4] = (9/4) * A_creep_3b[m4] * (1 - D[m4]) * P_eff[m4]

            return G, gamma
            

        
        previous_T = self.T.copy()
        previous_rho = self.rho.copy()  #Mg/m3
        T_s = self.T_surf[t + 1]        # °C, 다음 시점 표면 온도
        rho_ice = self.rho_ice[t + 1]
        D = previous_rho / rho_ice    # 0-dim
        D[D >= 1.0 - 1E-9] = 1.0 - 1E-9

        P = np.zeros(self.Nz)   #Pa
        for i in range(1, self.Nz):
            P[i] = np.trapz(previous_rho[:i + 1], self.z[:i + 1]) * C.g * C.Mg_to_kg
        
        # --- D_0: Stage 1→2 전환 밀도 (Eq. A2) ---
        D_0 = min(0.00226 * T_s + 0.647, 0.56)

        # gamma = 1.60228E-9
        gamma = 0.5 / C.year_to_sec
        # gamma = 7.47E-13

        # G, gamma = D_dot(D, P, gamma, previous_T, D_0, T_s)

        Dms  = D_0 + 0.009
        ind1 = np.argmax(D >= Dms)

        G, gamma = D_dot(D, P, gamma, previous_T, D_0, T_s)
        G_grad = np.gradient(G, D)
        G_grad[np.isnan(G_grad)] = 0.0

        ##############################################################
        debug = False
        if debug:
            fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)

            z_stage12 = self.z[ind1]
            z_stage23 = self.z[D > 0.9][0]  if np.any(D > 0.9)  else self.z[-1]
            z_stage3b = self.z[D > 0.98][0] if np.any(D > 0.98) else self.z[-1]

            for ax, x, xlabel in zip(axes,
                                    [G,  G_grad,          D,                     P],
                                    ['G (dD/dt) [s⁻¹]', 'G_grad [s⁻¹/(-)]', 'D (relative density)', 'P (overburden) [Pa]']):
                ax.plot(x, self.z, color='tab:blue')
                ax.axhline(z_stage12, color='tab:orange', linestyle='--', linewidth=0.8, label=f'Stage1→2 (D={D[ind1]:.3f})')
                ax.axhline(z_stage23, color='tab:green',  linestyle='--', linewidth=0.8, label='Stage2→3a (D=0.9)')
                ax.axhline(z_stage3b, color='tab:red',    linestyle='--', linewidth=0.8, label='Stage3a→3b (D=0.98)')
                ax.axhline(self.z_COD, color='k',         linestyle='--', linewidth=0.8, label='COD')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Depth [m]')
                ax.set_ylim(self.z[-1], self.z[0])
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(fontsize=6)

            fig.suptitle(f't = {self.t[t]:.2f} yr', fontsize=12)
            plt.show()
        ##################################################################


        # --- Newton-Raphson 반복 (3회) ---
        next_rho = previous_rho.copy()
        for _ in range(3):
            next_rho = self._newton_raphson_solve(t, G, G_grad, next_rho, previous_rho)

        return next_rho

    def _update_p_op(self, t):
        """
        Buizert & Severinghaus (2016), Eq. (8) — 피른 내 기압 전파 모델
        Backward Euler (fully implicit) 방법으로 풀이.
 
        ================================================================
        원본 PDE (논문 Eq. 8):
        ================================================================
 
            dp/dt = (p_bar / (mu * s_op)) * [ k* d²p/dz²
                                             + (dk*/dz - k* Mg/RT) dp/dz
                                             - (dk*/dz) (Mg/RT) p ]
 
        여기서:
            p(z,t)  = 개방 공극 내 기압 [Pa]
            p_bar   = 연평균 기압 (상수 근사) [Pa]
            mu      = 공기 동점성 계수 [Pa·s]
            s_op    = 개방 공극률 [-]
            k*      = k · s_op  (유효 투과도, k=피른 투과도) [m²]
            M       = 공기 몰질량 [kg/mol]
            g       = 중력가속도 [m/s²]
            R       = 기체상수 [J/(mol·K)]
            T       = 온도 [K]
 
        ================================================================
        정리하면 표준 형태:
        ================================================================
 
            dp/dt = A(z) d²p/dz² + B(z) dp/dz + C(z) p
 
        where:
            F   = p_bar / (mu * s_op)           ... 공통 계수
            A_z = F * k*                        ... 확산 계수
            B_z = F * (dk*/dz - k* Mg/RT)       ... 이류 계수
            C_z = F * (-dk*/dz * Mg/RT)         ... 반응 계수
 
        ================================================================
        Backward Euler 이산화:
        ================================================================
 
        (p_i^{n+1} - p_i^n) / dt = A_i (p_{i+1}^{n+1} - 2p_i^{n+1} + p_{i-1}^{n+1}) / dz²
                                 + B_i (p_{i+1}^{n+1} - p_{i-1}^{n+1}) / (2 dz)
                                 + C_i  p_i^{n+1}
 
        정리하면:
 
            -[A_i dt/dz² - B_i dt/(2dz)] p_{i-1}^{n+1}
          + [1 + 2 A_i dt/dz² - C_i dt]  p_i^{n+1}
          - [A_i dt/dz² + B_i dt/(2dz)]  p_{i+1}^{n+1}
          = p_i^n
 
        이를 삼중대각 행렬로 풀면:
            sub[i]  = -(A_i dt/dz² - B_i dt/(2dz))
            diag[i] = 1 + 2 A_i dt/dz² - C_i dt
            sup[i]  = -(A_i dt/dz² + B_i dt/(2dz))
 
        ================================================================
        경계조건:
        ================================================================
            z=0 (표면)   : p = p_atm(t)               (Dirichlet)
            z=Z (하단)   : dp/dz = 0                   (Neumann, zero-flux)
 
        ================================================================
        투과도 파라미터화 (Adolph & Albert, 2014):
        ================================================================
            k = 10^(-7.7 + 14.46 * s_op)   [m²], s_op > threshold
            k* = k * s_op
        """
        
        # --- 시간 스텝 [초 단위] ---
        dt_sec = C.dt * C.year_to_sec   # yr → s
 
        # --- 물리 상수 ---
        mu  = 1.81E-5        # Pa·s, 공기 점성 (~-20°C 근방 근사)
        mu  = 1.74E-5
        # --- 연평균 기압 p_bar ---
        p_bar = np.mean(self.p_atm)     # Pa
 
        # --- 투과도 (Adolph & Albert, 2014) ---
        k = np.power(10.0, -7.7 + 14.46 * self.s_op_safe[:self.M])   # m²
        # k[self.s_op < 1E-10] = 0.0

        k_star = k * self.s_op_safe[:self.M]                           # m²
 
        # --- dk*/dz (중심 차분) ---
        dk_star_dz = np.gradient(k_star, self.z[:self.M])
 
        # --- 계수 A(z), B(z), C(z) ---
        F = p_bar / (mu * self.s_op_safe[:self.M])  # [Pa / (Pa·s)] = [1/s]
 
        A_z = F * k_star                                   # m²/s
        B_z = F * (dk_star_dz - k_star * C.M_air * C.g / (C.R * self.T[:self.M]))  # m/s
        C_z = F * (-dk_star_dz * C.M_air * C.g / (C.R * (self.T[:self.M])))       # 1/s
 
        # --- Backward Euler 삼중대각 계수 ---
        dz_L = self.dz_L[:self.M]
        dz_R = self.dz_R[:self.M]
        dz_S = self.dz_S[:self.M]

        r_diff_L = A_z * dt_sec / (dz_L * dz_S)
        r_diff_C = A_z * dt_sec * (1.0/dz_L + 1.0/dz_R) / dz_S
        r_diff_R = A_z * dt_sec / (dz_R * dz_S) # 확산 무차원수
        r_adv  = B_z * dt_sec / (2.0 * dz_S)    # 이류 무차원수
        r_reac = C_z * dt_sec                  # 반응 무차원수
 
        sub  = -(r_diff_L - r_adv)               # 하삼각
        diag = 1.0 + r_diff_C - r_reac    # 대각
        sup  = -(r_diff_R + r_adv)               # 상삼각
 
        rhs = self.p_gas[:self.M].copy()                       # 우변 = p^n
 
        # --- 경계조건 적용 ---
        # z=0 : Dirichlet (표면 기압 = 대기압)
        p_surf = self.p_atm[t + 1]             # 다음 시점 대기압
        diag[0] = 1.0
        sup[0]  = 0.0
        sub[0]  = 0.0
        rhs[0]  = p_surf
 
        # z=Z : Neumann (dp/dz = 0 → p_{N-1} = p_{N-2})
        # backward: p_{N-1}^{n+1} - p_{N-2}^{n+1} = 0
        diag[-1] = 1.0
        sub[-1]  = -1.0
        sup[-1]  = 0.0
        rhs[-1]  = 0.0
 
        # --- Thomas 알고리즘 (삼중대각 직접 풀이) ---
        p_op = self._thomas_solve(sub, diag, sup, rhs)
        p_op[0] = self.p_atm[t + 1]
        p_gas = np.zeros(self.Nz)
        p_gas[:self.M] = p_op.copy()
        p_gas[self.M:] = p_gas[self.M - 1]

        return p_op, p_gas
    
    def _update_C_op(self, t, COD_idx):
        """
        Buizert (2011), Chapter 5, Eq.(5.16)-(5.23) — 개방 공극 내 트레이서 수송
        Crank-Nicolson implicit 방식으로 한 타임스텝 진행.
 
        ================================================================
        원본 PDE (논문 Eq. 5.16):
        ================================================================
 
            ∂C/∂t = α ∂²C/∂z² + β ∂C/∂z + γ C
 
        여기서 (Eq. 5.17-5.20):
            α   = D_X + D_eddy                                           [m²/s]
            β   = -D_X(ΔM·g/RT) + D_eddy(M_air·g/RT)
                  - w_air + (1/s*_op) d/dz[s*_op(D_X + D_eddy)]         [m/s]
            γ   = -(ΔM·g/RT) · (1/s*_op) d/dz[s*_op · D_X] - λ_X      [1/s]
 
        trapping rate θ는 γ 식에서 상쇄됨 (논문 Eq. 5.20 참조):
            θ가 ∂C/∂t 식의 -θC 항으로 들어가지만,
            γ의 d/dz[s*_op · w_air]/s*_op 항과 정확히 상쇄되어
            γ에는 θ 항이 남지 않음.
 
        ================================================================
        Crank-Nicolson 이산화 (논문 Eq. 5.21-5.23):
        ================================================================
 
            α*_i = α_i Δt / (2Δz²)
            β*_i = β_i Δt / (4Δz)
            γ*_i = γ_i Δt / 2
 
        A 행렬 (implicit, t+1 시점):
            sub[i]  = -(α*_i - β*_i)
            diag[i] = 1 + 2α*_i - γ*_i
            sup[i]  = -(α*_i + β*_i)
 
        B 행렬 (explicit, t 시점):
            sub[i]  = (α*_i - β*_i)
            diag[i] = 1 - 2α*_i + γ*_i
            sup[i]  = (α*_i + β*_i)
 
        풀이: A · C^{n+1} = B · C^n
 
        ================================================================
        경계조건 (논문 Eq. 5.27-5.28):
        ================================================================
            z=0 (표면)   : C = C_atm(t)             (Dirichlet)
            z=M·Δz (하단): C_M - C_{M-1} = 0        (Neumann, zero-gradient)
 
        ================================================================
        물리 상태:
        ================================================================
        _update_T, _update_rho, _update_p_op 이후 갱신된 self.T, self.rho,
        self.p_op를 사용하여 α, β, γ를 현재 타임스텝에 맞게 재계산.
        (단, D_X, D_eddy, w_air, s_op는 init 시 계산된 값 그대로 사용.
         완전한 동적 업데이트는 _update_rho/_update_p_op 연동 시 확장 가능.)
        """
        Delta_M = self.G.M_X - C.M_air
        Delta_t = C.dt * C.year_to_sec   # yr → s
        Delta_z = C.dz
 
        # --- 계수 α, β, γ (논문 Eq. 5.17-5.20) ---
        alpha = self.D_X + self.D_eddy                           # [m²/s]
 
        _beta = (1.0 / self.s_op_star[:self.M]
                 * np.gradient(self.s_op_star[:self.M] * (self.D_X + self.D_eddy), self.z[:self.M]))
        _beta[self.COD_idx] = _beta[self.COD_idx + 1]           # COD 특이점 보정
 
        beta = (-self.D_X * (Delta_M * C.g) / (C.R * self.T[:self.M])
                + self.D_eddy * (C.M_air * C.g) / (C.R * self.T[:self.M])
                - self.w_air[:self.M]                     # m/yr → m/s
                + _beta)                                          # [m/s]
 
        _gamma = (1.0 / self.s_op_star[:self.M]
                  * np.gradient(self.s_op_star[:self.M] * self.D_X, self.z[:self.M]))
        _gamma[self.COD_idx] = _gamma[self.COD_idx + 1]
 
        gamma = (-Delta_M * C.g / C.R / self.T[:self.M] * _gamma
                 - self.G.lambda_X)                               # [1/s]
 
        # --- 무차원 Crank-Nicolson 계수 (논문 Eq. 5.22) ---
        alpha_i_star = alpha * Delta_t / (2.0 * Delta_z ** 2)
        beta_i_star  = beta  * Delta_t / (4.0 * Delta_z)
        gamma_i_star = gamma * Delta_t / 2.0
 
        M = self.M - 1   # 유효 깊이 인덱스 상한
 
        # --- A 행렬 (암시적, t+1 시점) (논문 Eq. 5.23, 5.25) ---
        diag_A = 1.0 + 2.0 * alpha_i_star - gamma_i_star
        sub_A  = -(alpha_i_star - beta_i_star)
        sup_A  = -(alpha_i_star + beta_i_star)
 
        # BC: 표면 Dirichlet (Eq. 5.27) — diag=1, sup=0
        sub_A[0]  = 0.0
        diag_A[0] = 1.0
        sup_A[0]  = 0.0
        # BC: 하단 Neumann zero-gradient (Eq. 5.28) — sub=-1, diag=1
        sub_A[M]  = -1.0
        diag_A[M] =  1.0
        sup_A[M]  =  0.0
 
        # --- rhs = B · C^n 직접 계산 (행렬 생성 없이) ---
        # B의 대각 벡터: sub_B, diag_B, sup_B
        diag_B = 1.0 - 2.0 * alpha_i_star + gamma_i_star
        sub_B  = alpha_i_star - beta_i_star
        sup_B  = alpha_i_star + beta_i_star
 
        C_prev = self.C_gas.copy()
 
        rhs = np.empty(M + 1)
        rhs[0]   = diag_B[0] * C_prev[0] + sup_B[0] * C_prev[1]
        rhs[1:M] = (sub_B[1:M]  * C_prev[:M - 1]
                  + diag_B[1:M] * C_prev[1:M]
                  + sup_B[1:M]  * C_prev[2:M + 1])
        rhs[M]   = sub_B[M] * C_prev[M - 1] + diag_B[M] * C_prev[M]
 
        # BC 우변 적용
        rhs[0] = self.C_atm[t + 1]      # 표면 대기 경계조건 (Eq. 5.27)
        rhs[M] = 0.0                    # Neumann BC (Eq. 5.28)
 
        # --- Thomas 알고리즘으로 A · C^{n+1} = rhs 풀기 ---
        C_gas = self._thomas_solve(sub_A, diag_A, sup_A, rhs)
        C_gas[0] = self.C_atm[t + 1]          # Dirichlet 표면 경계 재확인

        C_op = C_gas.copy()
        C_op[COD_idx:] = 0.0    # COD 이하 개방공극 농도 = 0

        return C_op, C_gas

    def _update_rho_COD_bar(self, t):
        return 1 / (1 / self.rho_ice[t + 1] + 6.95E-4 * self.T_surf[t + 1] - 4.3E-2)    #Martinerie et al., 1994, Mitchell et al., 2015

    def _update_s(self, t, rho):
        return 1 - rho / self.rho_ice[t + 1]

    def _update_s_cl(self, t, s, rho_COD_bar):
        #################Goujon et al., 2003###############################
        s_co_bar = 1 - rho_COD_bar / self.rho_ice[t + 1]
        s_cl = 0.37 * s * np.power(s / s_co_bar, -7.6)
        s_cl[s_cl > s] = s[s_cl > s]

        return s_cl

    def _update_COD(self, rho, s_cl):
        COD_idx = np.argmax(s_cl)
        # self.rho_COD = 1 / (1 - 1 / 75) / (1 / C.kg_to_g / self.S.rho_ice + 7.02E-7 * self.S.T - 4.5E-5) / C.kg_to_g 
        # self.rho_COD = 75 / 74 / (1 / self.S.rho_ice + self.S.T * 6.95E-4 - 4.3E-2)
        rho_COD = rho[COD_idx]
        z_COD = self.z[COD_idx]
        return COD_idx, rho_COD, z_COD

    def _update_s_op(self, T, p_gas, s, s_cl, COD_idx):
        s_op = s - s_cl
        s_op[COD_idx:] = 0.0
        s_op_safe = s_op + 1E-9
        s_op_star = s_op_safe * (p_gas / C.P0) * (C.T0 / T)
        return s_op, s_op_safe, s_op_star

    def _update_p_cl(self, t, T, rho, p_gas, w_ice, s_cl):
        dt_sec = C.dt * C.year_to_sec

        # --- parcel의 이전 위치 ---
        z_prev = self.z - w_ice * dt_sec

        # --- T_ratio ---
        T_prev  = np.interp(z_prev, self.z, self.T, left=self.T_surf[t + 1])
        T_ratio = T / T_prev

        # --- eff_strain ---
        rho_prev   = np.interp(z_prev, self.z, self.rho, left=self.S.rho_0)
        eff_strain = -(rho - rho_prev) / rho_prev

        # --- ξ ---
        zeta = T_ratio / (1.0 + eff_strain)
        zeta = np.maximum(zeta, 1.0)

        # --- p_cl_adv: z_prev < 0이면 p_op[0] (표면 대기압) ---
        p_cl_prev = np.interp(z_prev, self.z, self.p_cl, left=self.p_atm[t + 1])

        # --- s_cl_prev: z_prev < 0이면 0 (표면에서 closed porosity 없음) ---
        s_cl_prev = np.interp(z_prev, self.z, self.s_cl, left=0.0)

        # --- 신규 트래핑 ---
        ds_cl_new = np.maximum(s_cl - s_cl_prev, 0.0)

        # --- p_cl 업데이트 ---
        num = p_cl_prev * zeta * s_cl_prev + p_gas * ds_cl_new
        den = s_cl_prev + ds_cl_new + 1E-30

        p_cl = num / den

        # 물리 제약
        p_cl = np.maximum(p_cl, p_gas)

        return p_cl

    def _update_iez(self, t, rho):
        iez = np.zeros(self.Nz)
        for i in range(1, self.Nz):
            iez[i] = np.trapz(rho[:i + 1], self.z[:i + 1])
        iez /= self.rho_ice[t + 1]
        return iez
    
    def _update_Xi(self, iez):
        Xi = 1 - iez / self.S.H
        return Xi

    def _update_w_ice(self, t, rho, iez):
        # --- Nye model ---
        # thinning: (1 - z_ice/H)
        w_ice = self.A_ieq[t + 1] * (self.rho_ice[t + 1] / rho) * (1.0 - iez / self.S.H)
        w_ice = np.maximum(w_ice, 0.0)   # 음수 방지
        return w_ice

    def _update_phi_op(self, s_op_star, w_air):
        return s_op_star * w_air

    def _update_phi_cl(self, T, s_cl, p_cl, w_ice):
        phi_cl = s_cl * (p_cl / C.P0) * (C.T0 / T) * w_ice
        return phi_cl

    def _update_w_air(self, T, Xi, w_ice, s_cl, p_cl, s_op_star, COD_idx):

        flux_COD = s_cl[COD_idx] * (p_cl[COD_idx] / C.P0) * w_ice[COD_idx] * (C.T0 / T[COD_idx]) * (Xi / Xi[COD_idx])
        flux_z = s_cl * (p_cl / C.P0) * w_ice * (C.T0 / T)
        w_air = (flux_COD + 1E-10 - flux_z) / (s_op_star + 1E-10)
        w_air = np.minimum(w_ice, w_air)
        w_air[COD_idx:] = w_ice[COD_idx:]

        return w_air

    def _update_C_cl(self, t, C_gas, w_ice, s_cl, p_gas, p_cl, s_op_star):
        """
        C_cl = Σ(Ci·Pi·s_cl_i) / Σ(Pi·s_cl_i)   (몰수 가중 평균)
        """
        dt_sec = C.dt * C.year_to_sec

        # C_gas (M+1,) → (Nz,) 패딩
        C_gas_full = np.zeros(self.Nz)
        C_gas_full[:len(C_gas)] = C_gas

        # --- parcel의 이전 위치 (Eulerian) ---
        z_prev = self.z - w_ice * dt_sec

        # --- advection ---
        C_cl_prev = np.interp(z_prev, self.z, self.C_cl,  left=self.C_atm[t + 1])
        s_cl_prev = np.interp(z_prev, self.z, self.s_cl,  left=0.0)
        p_cl_prev = np.interp(z_prev, self.z, self.p_cl,  left=self.p_atm[t + 1])

        # --- 신규 트래핑 ---
        ds_cl_new = np.maximum(s_cl - s_cl_prev, 0.0)

        # --- C_cl 업데이트 (몰수 = P·s_cl 가중) ---
        # 기존: Ci 불변, Pi→Pi·ξ, s_cl_i 불변 → Pi·s_cl_i = p_cl_prev·s_cl_prev·ξ
        # 신규: Ci=C_gas, Pi=p_op, s_cl_i=ds_cl_new
        n_old = p_cl_prev * s_cl_prev   # 기존 몰수 가중치 (ξ 상쇄: p_cl_next의 분모와 맞춤)
        n_new = p_gas * ds_cl_new   # 신규 몰수 가중치

        num = C_cl_prev * n_old + C_gas_full * n_new
        den = n_old + n_new

        C_cl = num / (den + 1E-30)

        # --- C_total ---
        full_air = s_op_star + s_cl * (p_cl / C.P0)
        C_total  = (C_cl * s_cl * (p_cl / C.P0) + C_gas_full * s_op_star) \
                / (full_air + 1E-30)

        return C_cl, C_total

    def _init_T_surf(self):
        T_surf = np.interp(self.t, self.S.T_surf[:, 0], self.S.T_surf[:, 1])
        return T_surf

    def _init_T_basal(self):
        T_basal = np.interp(self.t, self.S.T_basal[:, 0], self.S.T_basal[:, 1])
        return T_basal
    
    def _init_T(self):
        T = np.interp(self.z, [self.z[0], self.z[-1]], [self.T_surf[0], self.T_basal[0]])
        return T
        #Goujon et al., 2003

    def _init_rho_ice(self):
        T_surf = self.T_surf - C.T0
        rho_ice = 0.9165 - T_surf * 1.4438E-4 - T_surf ** 2 * 1.5175E-7
        return rho_ice

    def _init_A_ieq(self):
        A_ieq = np.interp(self.t, self.S.A_ieq[:, 0], self.S.A_ieq[:, 1])
        return A_ieq
    
    def _init_A_weq(self):
        A_weq = np.mean(self.A_ieq) / np.mean(self.rho_ice)
        return A_weq

    def _init_p_atm(self):
        p_atm = np.interp(self.t, self.S.p_atm[:, 0], self.S.p_atm[:, 1])
        return p_atm

    def _init_C_atm(self):
        C_atm = np.interp(self.t, self.S.C_atm[:, 0], self.S.C_atm[:, 1])
        return C_atm

    def _init_C_op(self):
        C_op = self.C_atm[0] * np.ones(self.M)
        C_gas = self.C_atm[0] * np.ones(self.M)
        C_op[self.COD_idx:] = 0.0
        return C_op, C_gas
    
    def _init_C_cl(self):
        C_cl = self.C_atm[0] * np.ones(self.Nz)
        C_total = self.C_atm[0] * np.ones(self.Nz)
        return C_cl, C_total
    #Buizert et al., 2016
    def _init_p_op(self):
        p_op = self.p_atm[0] * np.exp((C.M_air * C.g * self.z) / (C.R * self.T[0]))
        p_gas = p_op[:self.M].copy()
        p_op[self.COD_idx:] = np.nan
        return p_op, p_gas
    #Herron and Langway, 1980
    def _init_rho(self):
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

    def _init_rho_COD_bar(self):
        return 1 / (1 / self.rho_ice[0] + 6.95E-4 * self.T_surf[0] - 4.3E-2)    #Martinerie et al., 1994, Mitchell et al., 2015

    def _init_D_X_0(self):
        D_CO2_0 = 5.75E-10 * self.T[:self.M] ** 1.81 * (C.P0 / self.p_gas)
        # self.D_CO2_0 = 1.638946715E-05  # Buizert's MATLAB code
        D_X_0 = self.G.gamma_X * D_CO2_0
        return D_X_0

    def _init_s(self):
        return 1 - self.rho / self.rho_ice[0]

    def _init_s_cl(self):
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
        s_co_bar = 1 - self.rho_COD_bar / self.rho_ice[0]
        s_cl = 0.37 * self.s * np.power(self.s / s_co_bar, -7.6)
        s_cl[s_cl > self.s] = self.s[s_cl > self.s]

        return s_cl
        
    def _init_s_op(self):
        s_op = self.s - self.s_cl
        s_op[self.COD_idx:] = 0.0
        s_op_safe = s_op + 1E-9
        s_op_star = s_op_safe.copy()
        s_op_star[:self.COD_idx] = s_op_safe[:self.COD_idx] * (self.p_op[:self.COD_idx] / C.P0) * (C.T0 / self.T[:self.COD_idx])
        return s_op, s_op_safe, s_op_star
      
    def _init_COD(self):
        COD_idx = np.argmax(self.s_cl)
        # self.rho_COD = 1 / (1 - 1 / 75) / (1 / C.kg_to_g / self.S.rho_ice + 7.02E-7 * self.S.T - 4.5E-5) / C.kg_to_g 
        # self.rho_COD = 75 / 74 / (1 / self.S.rho_ice + self.S.T * 6.95E-4 - 4.3E-2)
        rho_COD = self.rho[COD_idx]
        z_COD = self.z[COD_idx]
        return COD_idx, rho_COD, z_COD
    
    def _init_rho_LID(self):
        return self.rho_COD - 0.01 #Kahel et al., 2021  #0.014    #Blunier et al., 2000

    def _init_z_LID(self):
        LID_idx = np.argmin(np.abs(self.rho - self.rho_LID))
        return self.z[LID_idx]

    def _init_iez(self):
        iez = np.zeros(self.Nz)
        for i in range(1, self.Nz):
            iez[i] = np.trapz(self.rho[:i + 1], self.z[:i + 1])
        iez /= self.rho_ice[0]
        return iez

    def _init_tau_inv_DZ(self):
        inv_tort = np.loadtxt(C.ROOT+"icecores\\"+self.S.name+"\\tortuosity_DZ.txt")
        inv_tort = np.interp(self.z, inv_tort[:, 0], inv_tort[:, 1])
        inv_tort[inv_tort > 1] = 1
        return inv_tort

    def _init_tau_inv_LIZ(self):
        D_m = np.loadtxt(C.ROOT+"icecores\\"+self.S.name+"\\tortuosity_LIZ.txt")
        D_m = np.interp(self.z, D_m[:, 0], D_m[:, 1])
        return D_m

    def _init_D_X(self):
        D_X = self.D_X_0 * self.tau_inv_DZ[:self.M]
        return D_X
    
    def _init_D_eddy(self):
        H = 4.5
        if self.S.name == "NEEM_EU":
            D_eddy_0 = 2.30453E-5
        elif self.S.name == "NEEM_US":
            D_eddy_0 = 2.426405E-5
        else:
            D_eddy_0 = 2.30453E-5
        # D_eddy_0 = 1.6E-5
        D_eddy = D_eddy_0 * np.exp(-self.z[:self.M] / H)
        D_eddy[self.z[:self.M] >= 55] = 0
        D_eddy = np.maximum(D_eddy, self.tau_inv_LIZ[:self.M])
        return D_eddy
    #수정해야될듯
    def _init_w_ice(self):
        #########Buizert et al., 2011###########
        # return self.S.A_ieq * self.S.rho_ice / self.rho
        #########Goujon et al., 2003###########
        zeta = self.z / self.S.H
        m = 10
        return self.rho_ice[0] / self.rho * (self.A_ieq[0] - self.A_ieq[0] * ((m + 2) / (m + 1) * zeta) * (1 - (zeta ** (m + 1)) / (m + 2)))

    def _init_p_cl(self):

        p_cl = np.zeros(self.Nz)
        dscl = np.gradient(self.s_cl, self.z)

        for i in range(self.COD_idx + 1):
            integral_num = []
            integral_den = []

            for j in range(i + 1):
                # ξ(z', z) = ρ(z')/ρ(z)  (무차원)
                zeta = self.rho[j] / self.rho[i]

                val_num = dscl[j] * self.p_gas[j] * (self.s[j] / self.s[i]) / zeta
                integral_num.append(val_num)
                integral_den.append(dscl[j])

            if np.sum(integral_den) != 0:
                p_cl[i] = (C.dz * np.sum(integral_num)) / (C.dz * np.sum(integral_den))
            else:
                p_cl[i] = self.p_gas[i]

        # COD 이후
        p_cl_z_COD = p_cl[self.COD_idx]
        for i in range(self.COD_idx + 1, self.Nz):
            zeta     = self.rho[self.COD_idx] / self.rho[i]
            p_cl[i] = p_cl_z_COD / zeta

        p_cl[0] = self.p_op[0]
        p_cl[:self.M]    = np.maximum(p_cl[:self.M], self.p_gas)

        return p_cl

    def _init_w_air(self):
        flux_COD = self.s_cl[self.COD_idx] * (self.p_cl[self.COD_idx] / C.P0) * self.w_ice[self.COD_idx] * (C.T0 / self.T[self.COD_idx]) * (self.Xi / self.Xi[self.COD_idx])
        flux_z = self.s_cl * (self.p_cl / C.P0) * self.w_ice * (C.T0 / self.T)
        w_air = (flux_COD + 1E-10 - flux_z) / (self.s_op_star + 1E-10)
        w_air = np.minimum(self.w_ice, w_air)
        w_air[self.COD_idx:] = self.w_ice[self.COD_idx:]
        return w_air
    
    def _init_Xi(self):
        Xi = 1 - self.iez / self.S.H
        return Xi

    def _init_phi_op(self):
        return self.s_op_star * self.w_air
    
    def _init_phi_cl(self):
        return self.s_cl * (self.p_cl / C.P0) * (C.T0 / self.T) * self.w_ice
    ##############CIC Model######################
    def _init_x_air(self):
        return (self.s_cl[self.COD_idx] * self.p_cl[self.COD_idx] * C.T0) / (self.T_surf[0] * C.P0 * self.rho[self.COD_idx])

    def _init_eta(self):
        eta = np.zeros(self.Nz)
        for i in range(0, self.Nz):
            eta[i] = np.trapz(1 / self.w_ice[:i + 1], self.z[:i + 1])
        return eta

    def _plot_axes(self, axes):
        z = self.z

        C_op_plot = np.full(self.Nz, np.nan)
        C_op_plot[:self.COD_idx] = self.C_op[:self.COD_idx]

        p_op_plot = np.full(self.Nz, np.nan)
        p_op_plot[:self.COD_idx] = self.p_op[:self.COD_idx]

        def _plot(ax, lines, xlabel):
            xlim = ax.get_xlim()
            is_first = not ax.lines  # 아직 아무것도 그려지지 않은 상태
            ax.cla()
            for x, label, color in lines:
                ax.plot(x, z, label=label, color=color)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Depth [m]")
            ax.set_ylim(z[-1], z[0])

            if not is_first:
                data_min = min(np.nanmin(x) for x, _, _ in lines)
                data_max = max(np.nanmax(x) for x, _, _ in lines)
                new_xlim = (min(xlim[0], data_min), max(xlim[1], data_max))
                ax.set_xlim(new_xlim)

            ax.axhline(self.z_COD, color="k",    linestyle="--", linewidth=0.8, label="COD")
            ax.axhline(self.z_LID, color="gray", linestyle="--", linewidth=0.8, label="LID")
            ax.legend(fontsize=6, loc="lower right")
            ax.grid(True, linestyle="--", alpha=0.5)

        _plot(axes[0],
            [(self.T,   "T",   "tab:red")],
            "T [K]")

        _plot(axes[1],
            [(self.rho, "ρ",   "tab:brown")],
            "ρ [Mg/m³]")

        _plot(axes[2],
            [(p_op_plot, "p_op", "tab:blue"),
            (self.p_cl, "p_cl", "tab:orange")],
            "Pressure [Pa]")

        _plot(axes[3],
            [(self.s_op, "s_op", "tab:green"),
            (self.s_cl, "s_cl", "tab:purple")],
            "Porosity [-]")

        _plot(axes[4],
            [(self.w_ice, "w_ice", "tab:blue"),
            (self.w_air, "w_air", "tab:cyan")],
            "Velocity [m/yr]")

        _plot(axes[5],
            [(self.phi_op, "φ_op", "tab:blue"),
            (self.phi_cl, "φ_cl", "tab:orange")],
            "Flux [m/s]")

        _plot(axes[6],
            [(C_op_plot,    "C_op",    "tab:blue"),
            (self.C_cl,    "C_cl",    "tab:orange"),
            (self.C_total, "C_total", "tab:green")],
            "Concentration [ppm]")

    def plot_state(self, title="", t=None):
        if not hasattr(self, '_fig') or not plt.fignum_exists(self._fig.number):
            self._fig, self._axes = plt.subplots(1, 7, figsize=(18, 8), sharey=True)
        self._fig.suptitle(title, fontsize=13)
        self._plot_axes(self._axes)
        plt.tight_layout()

        # boundary condition 수직선 업데이트
        if t is not None and hasattr(self, '_bc_vlines'):
            for vline in self._bc_vlines:
                vline.set_xdata([t, t])
            self._bc_fig.canvas.draw_idle()

        plt.pause(0.01)

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

    def plotDage(self):
        plt.gca().invert_yaxis()
        plt.ylabel("Depth [m]")
        plt.xlabel("Delta Age [yr]")
        plt.plot(self.Delta_age, self.z[self.COD_idx:self.M + 1], c="r", marker="^", label="Delta Age")
        plt.legend()
        plt.grid(True)
        _, xmax = plt.xlim()
        # plt.xlim(0, xmax * 1.5)
        plt.xlim(0, 40)
        plt.show()

    def compareCH4(self):
        year_CH4 = np.loadtxt(C.ROOT+"icecores\\CPSW\\year_CH4.txt")
        depth_CH4_measured = np.loadtxt(C.ROOT+"icecores\\CPSW\\CH4_raw.txt")
        year = self.S.sample_year - self.t
        CH4_measured = np.interp(year, year_CH4[:, 0], year_CH4[:, 1])
        depth_CH4_modelled = self.C_cl @ CH4_measured
        depth_CH4_modelled *= C.dt

        gamma_years = list()
        for i in range(self.Nz):
            gammas, _ = self._Delta(self.z[i])
            gamma_years.append(self.S.sample_year - gammas[1])
        gamma_years = np.array(gamma_years)

        depth_CH4_modelled_gamma = np.interp(gamma_years, year_CH4[:, 0], year_CH4[:, 1])
        plt.errorbar(depth_CH4_measured[:, 1], depth_CH4_measured[:, 0], xerr=depth_CH4_measured[:, 2], c="r", ecolor="black", label="Measured")
        plt.plot(depth_CH4_modelled, self.z, c="b", label="Modelled (GAD)")
        plt.plot(depth_CH4_modelled_gamma, self.z, c="g", label="Modelled (Mean)")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend()
        plt.xlabel("CH4 [ppb]")
        plt.ylabel("Depth [m]")
        plt.show()

    def plot_boundary_conditions(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Boundary Conditions", fontsize=13)

        axes[0, 0].plot(self.t, self.T_surf,  color="tab:red")
        axes[0, 0].set_xlabel("t [yr]")
        axes[0, 0].set_ylabel("T_surf [°C]")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        axes[0, 1].plot(self.t, self.A_ieq,   color="tab:blue")
        axes[0, 1].set_xlabel("t [yr]")
        axes[0, 1].set_ylabel("Accumulation [m/s]")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        axes[1, 0].plot(self.t, self.p_atm,   color="tab:green")
        axes[1, 0].set_xlabel("t [yr]")
        axes[1, 0].set_ylabel("p_atm [Pa]")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        axes[1, 1].plot(self.t, self.C_atm,   color="tab:orange")
        axes[1, 1].set_xlabel("t [yr]")
        axes[1, 1].set_ylabel("C_atm [ppm]")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        # 현재 시간 수직선 저장
        self._bc_vlines = [
            ax.axvline(self.t[0], color="k", linestyle="--", linewidth=1.0)
            for ax in axes.flat
        ]
        self._bc_fig = fig

        plt.tight_layout()
        plt.pause(0.01)