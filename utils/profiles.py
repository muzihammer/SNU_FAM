import utils.constants as C

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

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

        # self.D_CO2_0 = 1.39E-5 * (self.S.T/C.C_to_K) ** 1.75 * (C.P0 / self.S.p_0)
        self.D_CO2_0 = 5.75E-10 * self.S.T ** 1.81 * (C.P0 / self.S.p_0)
        # self.D_CO2_0 = 1.638946715E-05  # Buizert's MATLAB code
        self.D_X_0 = self.G.gamma_X * self.D_CO2_0
        
        #######Goujon et al., 2003#########
        self.T_surf = self._init_T_surf()   #[1xNt]
        self.T_basal = self._init_T_basal() #[1xNt]
        self.T = self._init_T()             #[NzX1]

        self.rho_ice = self._init_rho_ice() 

        self.A_ieq = self._init_A_ieq()     #[1xNt]
        self.A_weq = self._init_A_weq()     #[1x1], for H-L model

        #######Buizert et al., 2016#########
        self.p_atm = self._init_p_atm()     #[1xNt]
        self.p_op = self._init_p_op()       #[Nzx1]

        #######Herron and Langway, 1980#########
        self.rho = self._init_rho() #[Nzx1], Mg/m3

        ##############Mitchell et al., 2015##############
        self.rho_COD_bar = self._init_rho_COD_bar()

        self.s = self._init_s()
        self.s_cl = self._init_s_cl()

        self.COD_idx = self._init_COD_idx()
        self.rho_COD = self._init_rho_COD()
        self.z_COD = self._init_z_COD()

        self.s_op, self.s_op_safe = self._init_s_op()
        self.s_op_star = self._init_s_op_star()


        self.iez = self._init_iez()
        self.rho_LID = self._init_rho_LID()
        self.z_LID = self._init_z_LID()

        self.Nz_gas = np.argmin(np.abs(self.z - (self.z_COD + 10)))
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
        for i, t in tqdm(enumerate(self.t)):
            T = self._update_T(i)
            rho = self._update_rho(i)
            p_op = self._update_p_op(i)                                        

            self.T = T
            self.rho = rho
            self.p_op = p_op


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
        
        def _thomas_solve(a, b, c, d):
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
        
        dt_sec = C.dt * C.year_to_sec

        previous_T = self.T.copy()
        previous_rho = self.rho.copy()
        rho_ice = self.rho_ice[t + 1]

        K_ice = 2.22 * (1 - 0.0067 * previous_T)   # Wm-1K-1
        K = (K_ice * (previous_rho / rho_ice) ** (2.0 - 0.5 * previous_rho / rho_ice)) # Wm-1K-1

        c_ice = (152.5 + 7.122 * (previous_T + C.C_to_K)) #Jkg-1K-1
        crho_firn = c_ice * previous_rho * (C.m_to_cm ** 3) / C.kg_to_g            #JK-1m-3

        K_half_temp = 0.5 * (K[:-1] + K[1:])       # K_{i+1/2}, [Nz-1]
        K_half_L = np.empty(self.Nz)             # K_{i-1/2}, [Nz]
        K_half_R = np.empty(self.Nz)        # [Nz]로 확장
        K_half_L[1:] = K_half_temp
        K_half_L[0]  = K[0]
        K_half_R[:-1] = K_half_temp
        K_half_R[-1]  = K[-1]

        w = self.w_ice / C.year_to_sec

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
        T_surf = self.T_surf[t + 1]
        diag[0] = 1.0
        sup[0]  = 0.0
        sub[0]  = 0.0
        rhs[0]  = T_surf

        # z=H : Dirichlet (바닥 온도)
        T_basal = self.T_basal[t + 1]
        diag[-1] = 1.0
        sub[-1]  = 0.0
        sup[-1]  = 0.0
        rhs[-1]  = T_basal

        # --- Thomas 알고리즘 ---
        next_T = _thomas_solve(sub, diag, sup, rhs)

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
            Q = 60E3
            
            # --- Stage 1: Snow, D < D_0 (Eq. A1) ---
            m1 = D < D_0
            G[m1] = gamma * (P[m1] / D[m1] ** 2) * (1 - (5 / 3) * D[m1])

            # --- Stage 2: Firn, 0.6 ≤ D < 0.895 (Eq. A3–A9) ---
            m2 = (0.6 <= D) & (D < 0.895)

            c_Z = 15.5
            Z_0 = 110.2 * D_0 ** 3 - 148.594 * D_0 ** 2 + 87.6166 * D_0 - 17

            l_prime = (D[m2] / D_0) ** (1 / 3)
            Z = Z_0 + c_Z * (l_prime - 1)

            l_2prime = l_prime + (
                4 * Z_0 * (l_prime - 1) ** 2 * (2 * l_prime + 1)
                + c_Z * (l_prime - 1) ** 3 * (3 * l_prime + 1)
            ) / (
                12 * l_prime * (4 * l_prime - 2 * Z_0 * (l_prime - 1) - c_Z * (l_prime - 1) ** 2)
            )

            a_contact = (np.pi / 3 / Z / l_prime ** 2) * (
                3 * (l_2prime ** 2 - 1) * Z_0
                + l_2prime ** 2 * c_Z * (2 * l_2prime - 3)
                + c_Z
            )

            A_creep = 7.89E-15 * np.exp(-Q / C.R / (previous_T[m2] + C.C_to_K))
            P_star = 4 * np.pi * P[m2] / a_contact / Z / D[m2]

            G[m2] = 5.3 * A_creep * (D[m2] ** 2 * D_0) ** (1 / 3) \
                    * (a_contact / np.pi) ** (1 / 2) * (P_star / 3) ** n

            # --- 전환 구간: Stage 1 → 2 (D_0 ≤ D < 0.6) ---
            m1_2 = (D_0 <= D) & (D < 0.6)
            if np.any(m1) and np.any(m2) and np.any(m1_2):
                hermite = CubicHermiteSpline(
                    x=[D[m1][-1], D[m2][0]],
                    y=[G[m1][-1], G[m2][0]],
                    dydx=[np.gradient(G[m1], D[m1])[-1], np.gradient(G[m2], D[m2])[0]]
                )
                G[m1_2] = hermite(D[m1_2])

            # --- Stage 3a: Bubbly ice, cylindrical, 0.905 ≤ D < 0.945 (Eq. A10) ---
            m3 = (0.905 <= D) & (D < 0.945)

            A_creep = 7.89E-15 * np.exp(-Q / C.R / (previous_T[m3] + C.C_to_K))
            P_atm = self.p_atm[t]
            V_c = (6.95E-4 * (T_s + C.C_to_K) - 0.0043) / C.m_to_cm ** 3 * C.kg_to_g
            D_c = 1 / (V_c * C.rho_ice * C.Mg_to_kg + 1)
            P_b = P_atm * (D[m3] * (1 - D_c) / D_c / (1 - D[m3]))
            P_eff = np.maximum(P[m3] + P_atm - P_b, 0.0)

            G[m3] = 2 * A_creep \
                    * (D[m3] * (1 - D[m3]) / (1 - (1 - D[m3]) ** (1 / n)) ** n) \
                    * (2 * P_eff / n) ** n

            # --- 전환 구간: Stage 2 → 3a (0.895 ≤ D < 0.905) ---
            m2_3 = (0.895 <= D) & (D < 0.905)
            if np.any(m2) and np.any(m3) and np.any(m2_3):
                hermite = CubicHermiteSpline(
                    x=[D[m2][-1], D[m3][0]],
                    y=[G[m2][-1], G[m3][0]],
                    dydx=[np.gradient(G[m2], D[m2])[-1], np.gradient(G[m3], D[m3])[0]]
                )
                G[m2_3] = hermite(D[m2_3])

            # --- Stage 3b: Bubbly ice, spherical, 0.955 ≤ D < 1.0 (Eq. A13) ---
            m4 = (0.955 <= D) & (D <= 1.0)

            A_creep = 1.2E-3 * np.exp(Q / C.R / previous_T[m4])
            P_b = P_atm * (D[m4] * (1 - D_c) / D_c / (1 - D[m4]))
            P_eff = np.maximum(P[m4] + P_atm - P_b, 0.0)

            G[m4] = (9 / 4) * A_creep * (1 - D[m4]) * P_eff

            # --- 전환 구간: Stage 3a → 3b (0.945 ≤ D < 0.955) ---
            m3_4 = (0.945 <= D) & (D < 0.955)
            if np.any(m3) and np.any(m4) and np.any(m3_4):
                hermite = CubicHermiteSpline(
                    x=[D[m3][-1], D[m4][0]],
                    y=[G[m3][-1], G[m4][0]],
                    dydx=[np.gradient(G[m3], D[m3])[-1], np.gradient(G[m4], D[m4])[0]]
                )
                G[m3_4] = hermite(D[m3_4])

            return G

        def _newton_raphson_solve(G, G_grad, next_rho, previous_rho):
            """
            Newton-Raphson 1회 sweep (위→아래 순차).

            F(ρ)  = (ρ − ρ^n)/dt_sec + w(ρ)(ρ − ρ_{i-1})/dz_L − ρ_ice · G(D)
            F'(ρ) = 1/dt_sec + w_ice·ρ_ice·ρ_{i-1}/(ρ²·dz_L) − dG/dD
            """
            dt_sec = C.dt * C.year_to_sec
            rho_ice = self.rho_ice[t + 1]
            w = self.w_ice / C.year_to_sec      # m/yr → m/s

            for j in range(1, len(next_rho)):
                rho_j = next_rho[j]
                rho_j_upper = next_rho[j - 1]

                w_j = w[j] * rho_ice / rho_j

                F = (rho_j - previous_rho[j]) / dt_sec \
                    + w_j * (rho_j - rho_j_upper) / self.dz_L[j] \
                    - rho_ice * G[j]

                F_prime = 1 / dt_sec \
                    + (w[j] * rho_ice * rho_j_upper) / (rho_j ** 2 * self.dz_L[j]) \
                    - G_grad[j]

                next_rho[j] = rho_j - F / F_prime

            return next_rho
        
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
        D_0 = min(0.00226 * T_s + 0.647, 0.59)

        # gamma = 1.60228E-9
        # gamma = 0.5 / C.year_to_sec
        gamma = 7.47E-13

        G = D_dot(D, P, gamma, previous_T, D_0, T_s)
        G_grad = np.gradient(G, D)
        G_grad[np.isnan(G_grad)] = 0.0
        
        # --- Newton-Raphson 반복 (3회) ---
        next_rho = previous_rho.copy()
        for _ in range(3):
            next_rho = _newton_raphson_solve(G, G_grad, next_rho, previous_rho)

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
        
        def _thomas_solve(a, b, c, d):
            """
            삼중대각 연립방정식 풀이 (Thomas algorithm / TDMA).
                a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]
        
            a[0], c[-1]은 사용되지 않음.
        
            Parameters
            ----------
            a : ndarray [N]  하삼각 (sub-diagonal)
            b : ndarray [N]  대각 (diagonal)
            c : ndarray [N]  상삼각 (super-diagonal)
            d : ndarray [N]  우변 (RHS)
        
            Returns
            -------
            x : ndarray [N]  해
            """
            N = len(d)
            # 복사 (원본 보존)
            c_ = c.astype(float).copy()
            d_ = d.astype(float).copy()
            b_ = b.astype(float).copy()
        
            # Forward sweep
            for i in range(1, N):
                m = a[i] / b_[i - 1]
                b_[i] -= m * c_[i - 1]
                d_[i] -= m * d_[i - 1]
        
            # Back substitution
            x = np.empty(N)
            x[-1] = d_[-1] / b_[-1]
            for i in range(N - 2, -1, -1):
                x[i] = (d_[i] - c_[i] * x[i + 1]) / b_[i]
        
            return x

        # --- 시간 스텝 [초 단위] ---
        dt_sec = C.dt * C.year_to_sec   # yr → s
 
        # --- 물리 상수 ---
        mu  = 1.81E-5        # Pa·s, 공기 동점성 (~-20°C 근방 근사)
 
        # --- 연평균 기압 p_bar ---
        p_bar = np.mean(self.p_atm)     # Pa
 
        # --- 투과도 (Adolph & Albert, 2014) ---
        k = np.power(10.0, -7.7 + 14.46 * self.s_op_safe)   # m²
        k[self.s_op < 1E-10] = 0.0
 
        k_star = k * self.s_op_safe                            # m²
 
        # --- dk*/dz (중심 차분) ---
        dk_star_dz = np.gradient(k_star, self.z)
 
        # --- 계수 A(z), B(z), C(z) ---
        F = p_bar / (mu * self.s_op_safe)  # [Pa / (Pa·s)] = [1/s]
 
        A_z = F * k_star                                   # m²/s
        B_z = F * (dk_star_dz - k_star * C.M_air * C.g / (C.R * self.T))  # m/s
        C_z = F * (-dk_star_dz * C.M_air * C.g / (C.R * self.T))          # 1/s
 
        # --- Backward Euler 삼중대각 계수 ---
        r_diff_L = A_z * dt_sec / (self.dz_L * self.dz_S)
        r_diff_C = A_z * dt_sec * (1.0/self.dz_L + 1.0/self.dz_R) / self.dz_S
        r_diff_R = A_z * dt_sec / (self.dz_R * self.dz_S) # 확산 무차원수
        r_adv  = B_z * dt_sec / (2.0 * self.dz_S)    # 이류 무차원수
        r_reac = C_z * dt_sec                  # 반응 무차원수
 
        sub  = -(r_diff_L - r_adv)               # 하삼각
        diag = 1.0 + r_diff_C - r_reac    # 대각
        sup  = -(r_diff_R + r_adv)               # 상삼각
 
        rhs = self.p_op.copy()                       # 우변 = p^n
 
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
        p_next = _thomas_solve(sub, diag, sup, rhs)
 
        return p_next
    
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
        rho_ice = 0.9165 - self.T_surf * 1.4438E-4 - self.T_surf ** 2 * 1.5175E-7
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

    #Buizert et al., 2016
    def _init_p_op(self):
        p_op = self.p_atm[0] * np.exp((C.M_air * C.g * self.z) / (C.R * self.T[0]))
        return p_op
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
        return 1 / (1 / self.S.rho_ice + 6.95E-4 * self.S.T - 4.3E-2)    #Martinerie et al., 1994, Mitchell et al., 2015

    def _init_s(self):
        return 1 - self.rho / self.S.rho_ice

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
        s_co_bar = 1 - self.rho_COD_bar / self.S.rho_ice
        s_cl = 0.37 * self.s * np.power(self.s / s_co_bar, -7.6)
        s_cl[s_cl > self.s] = self.s[s_cl > self.s]

        return s_cl
        
    def _init_s_op(self):
        s_op = self.s - self.s_cl
        s_op[self.COD_idx:] = 0.0
        s_op_safe = s_op + 1E-9
        return s_op, s_op_safe
    
    def _init_s_op_star(self):
        return self.s_op * np.exp(C.M_air * C.g * self.z / C.R / self.S.T)
        # return self.s_op * self.rho / self.rho_LID
      
    def _init_COD_idx(self):
        return np.argmax(self.s_cl)

    def _init_rho_COD(self):
        # self.rho_COD = 1 / (1 - 1 / 75) / (1 / C.kg_to_g / self.S.rho_ice + 7.02E-7 * self.S.T - 4.5E-5) / C.kg_to_g 
        # self.rho_COD = 75 / 74 / (1 / self.S.rho_ice + self.S.T * 6.95E-4 - 4.3E-2)
        return self.rho[self.COD_idx]
    
    def _init_z_COD(self):
        return self.z[self.COD_idx]
    
    def _init_rho_LID(self):
        return self.rho_COD - 0.01 #Kahel et al., 2021  #0.014    #Blunier et al., 2000

    def _init_z_LID(self):
        LID_idx = np.argmin(np.abs(self.rho - self.rho_LID))
        return self.z[self.LID_idx]

    def _init_iez(self):
        iez = np.zeros(self.Nz)
        for i in range(1, self.Nz):
            iez[i] = np.trapz(self.rho[:i + 1], self.z[:i + 1])
        iez /= self.S.rho_ice
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
        D_X = self.D_X_0 * self.tau_inv_DZ
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
        D_eddy = D_eddy_0 * np.exp(-self.z / H)
        D_eddy[self.z >= 55] = 0
        D_eddy = np.maximum(D_eddy, self.tau_inv_LIZ)
        return D_eddy

    #수정해야될듯
    def _init_w_ice(self):
        #########Buizert et al., 2011###########
        # return self.S.A_ieq * self.S.rho_ice / self.rho
        #########Goujon et al., 2003###########
        zeta = self.z / self.S.H
        m = 10
        return self.S.rho_ice / self.rho * (self.S.A_ieq - self.S.A_ieq * ((m + 2) / (m + 1) * zeta) * (1 - (zeta ** (m + 1)) / (m + 2)))

    def _init_p_cl(self):

        p_cl = np.zeros(self.Nz)
        # dz = self.z[1] - self.z[0] # dz (단일 값) 추출
        
        strain = np.gradient(np.log(self.w_ice), self.z)

        dscl = np.gradient(self.s_cl, self.z)
        # dscl = np.concatenate([[0], np.diff(self.s_cl) / C.dz])
        exp_term = np.exp(C.M_air * C.g * self.z / C.R / self.S.T)

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

    def _init_w_air(self):
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
    
    def _init_phi_op(self):
        return self.s_op_star * self.w_air
    
    def _init_phi_cl(self):
        return self.s_cl * self.p_cl * self.w_ice

    ##############CIC Model######################
    def _init_x_air(self):
        return (self.s_cl[self.COD_idx] * self.p_cl[self.COD_idx] * C.C_to_K) / (self.S.T * C.P0 * self.rho[self.COD_idx])

    def _init_eta(self):
        eta = np.zeros(self.Nz)
        for i in range(0, self.Nz):
            eta[i] = np.trapz(1 / self.w_ice[:i + 1], self.z[:i + 1])
        return eta
    
    def _init_C_op(self):
        Delta_M = self.G.M_X - C.M_air
        Delta_t = C.dt * C.year_to_sec
        Delta_z = C.dz

        alpha = self.D_X + self.D_eddy
        _beta = 1 / self.s_op_star * np.gradient(self.s_op_star * (self.D_X + self.D_eddy), self.z)
        _beta[self.COD_idx] = _beta[self.COD_idx + 1]
        beta = -self.D_X * (Delta_M * C.g) / (C.R * self.S.T) + self.D_eddy * (C.M_air * C.g) / (C.R * self.S.T) - (self.w_air) / (C.year_to_sec) + _beta
        _gamma = 1 / self.s_op_star * np.gradient(self.s_op_star * self.D_X, self.z)
        _gamma[self.COD_idx] = _gamma[self.COD_idx + 1]
        gamma = -Delta_M * C.g / C.R / self.S.T * _gamma - self.G.lambda_X

        alpha_i_star = alpha * Delta_t / (2 * Delta_z * Delta_z)
        beta_i_star = beta * Delta_t / (4 * Delta_z)
        gamma_i_star = gamma * Delta_t / 2


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

        C_atm = np.zeros(self.Nt + 1)
        C_atm[(0.2 - 1E-9 <= self.t) & (self.t < 0.4 - 1E-9)] = 1 / (0.4 - 0.2)
        C_op = np.zeros(self.C_shape)

        C_op[:, 0] = C_atm[0]
        BC_n = np.zeros(self.M + 1)
        for t in tqdm(range(self.Nt)):
            C_prev = C_op[:self.M + 1, t]
            BC_n = B @ C_prev
            
            BC_n[0] = C_atm[t]
            BC_n[self.M] = 0
            
            C_next = A_inv @ BC_n
            C_next[0] = C_atm[t + 1]
            # C_next[C_next < 0] = 0
            C_op[:self.M + 1, t + 1] = C_next.flatten()
            C_op[self.M + 1:, t + 1] = C_op[self.M, t + 1]

        C_gas = C_op.copy()
        C_op[self.COD_idx:, :] = 0.0
        # C_op_test = np.sum(C_op, axis = 1)
        return C_op, C_gas
    
    def _init_C_cl(self):
        trapping_t = np.gradient(self.phi_cl, self.z) / self.w_ice

        C_cl = np.zeros(self.C_shape)
        C_total = np.zeros(self.C_shape)
        M_position = np.zeros(self.C_shape)
        M_C = np.ones(self.C_shape)
        M_position[:, -1] = self.z.copy()
        for i in range(1, self.Nt + 1):
            M_position[:, self.Nt - i] = np.interp(self.eta - i * C.dt, np.concatenate(([-100000, -self.eta[1]], self.eta)), np.concatenate(([-10, -C.dz], self.z)))
        M_position[M_position < 0] = -10
        M_C[M_position < 0] = 0
        
        M_trapping = np.interp(M_position, np.concatenate(([-10, -C.dz], self.z)), np.concatenate(([0, trapping_t[0]], trapping_t)))

        full_air = self.s_op_star + self.s_cl * self.p_cl

        for i in tqdm(range(0, self.Nt + 1)):
            M_C_open = np.zeros(self.C_shape)
            M_C_open[:, self.Nt] = self.C_gas[:, self.Nt - i]
            for j in range(1, self.Nt - i + 1):
                M_C_open[:, self.Nt - j] = np.interp(M_position[:, self.Nt - j], np.concatenate(([-10, -C.dz], self.z)), np.concatenate(([1, self.C_gas[0, self.Nt - j - i]], self.C_gas[:, self.Nt - j - i])))
            C_cl[:, self.Nt - i] = np.sum(M_C_open * M_trapping, axis=1) / np.sum(M_C * M_trapping, axis=1)
            C_total[:, self.Nt - i] = (C_cl[:, self.Nt - i] * self.s_cl * self.p_cl + self.C_gas[:, self.Nt - i] * self.s_op_star) / full_air
        
        # C_cl_test = np.sum(C_cl, 1)

        return C_cl, C_total

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

