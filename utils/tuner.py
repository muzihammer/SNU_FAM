import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import utils.constants as C
from utils.site import Site
from utils.gas import Gas
from utils.profiles import Profile

def random_fourier_curve(n, K=8, sigma=0.3, p=2.0, x=None, rng=None):
    rng = np.random.default_rng(rng)
    if x is None:
        x = np.linspace(0, n, 500)
    a = rng.normal(0, sigma, size=K) / (np.arange(1, K+1)**p)
    base = 1 - x/n
    wiggle = sum(a[k-1]*np.sin(k*np.pi*x/n) for k in range(1, K+1))
    return x, base + wiggle  # f(0)=1, f(n)=0

class Tuner:
    def __init__(self):
        df_sites = pd.read_excel(C.ROOT+"sites.xlsx")
        self.sites = list()
        for i, site in df_sites.iterrows():
            self.sites.append(Site(site["Site"], site["T[C]"], site["A[m]"], site["p[hPa]"], site["Year"], site["rho_0"], site["use_HL"] == "O"))

    def run(self):
        query_1 = "Tortuosity 튜닝을 원하는 지역을 선택하라냥 =^._.^= ∫\n"
        for i, data in enumerate(self.sites):
            query_1 += str(i + 1) + ". " + data.name + "\n"
        query_1 += ":\t"
        id_1 = int(input(query_1)) - 1
        site = self.sites[id_1]

        with open(file='E:\\LICP\\code\\SNUmodel\\results\\'+site.name+'.pkl', mode='rb') as f:
            P = pickle.load(f)


        CH4_target = np.loadtxt("E:\\LICP\\code\\data\\SNUmodel\\icecores\\CPSW\\CH4_raw.txt")
        CH4_gt = np.loadtxt("E:\\LICP\\code\\data\\year_CH4.txt") #year - CH4


        
        s, s_cl, s_op, rho = P.porosity()
        w_air, w_ice, p = P.velocity()
        z = P.z
        D_X_0 = 5.75E-10 * site.T ** 1.81 * (C.P0 / site.p_0)

        g1 = -0.209
        g2 = 1.515
        g3 = 0.53
        g4 = 3.17E-10
        g5 = 1.82
        best_params = [g1, g2, g3, g4, g5]
        # s_op = self.s_op_star[:self.LID_idx + 1]
        D_X_before_LID = D_X_0 * (g1 + g2 * s_op + g3 * s_op * s_op)
        LID_idx = np.min(np.where(D_X_before_LID < 0))
        D_X_before_LID = D_X_before_LID[:LID_idx]
        # D_CH4_before_LID[D_CH4_before_LID < 0] = 0

        rho_LID = rho[LID_idx]
        z_LID = z[LID_idx]

        D_X_after_LID = g4 + (D_X_before_LID[-1] - g4) * np.exp(-g5 * (z[LID_idx:] - z_LID))
        
        D_X = np.concatenate((D_X_before_LID, D_X_after_LID))
        
        D_X_best = D_X
        RMSE = np.inf

        epsilon = 10

        plt.ion()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        start_depth = CH4_target[0, 0]
        end_depth = CH4_target[-1, 0]
        mid_depth = (start_depth + end_depth) / 2
        mid_depth = (start_depth + mid_depth) / 2
        # mid_depth = (start_depth + mid_depth) / 2
        CH4_target = np.interp(z, CH4_target[:, 0], CH4_target[:, 1])
        axs[0].plot(z, D_X_best)
        axs[1].plot(z, CH4_target)
        count = 0
        while(count < 100):


            # plt.plot(z, D_X)
            # plt.show()

            # inv_tort = D_X / D_X_0 # / s_op
            # plt.plot(z, inv_tort)
            # plt.show()

            
            g6 = 3.17E-9
            g7 = 0.11
            H = 4.5
            N = P.N
            COD_idx = P.COD_idx

            #########################SIO Model#####################################
            D_eddy_before_LID = 1.6E-5 * np.exp(-z[:LID_idx + 1] / H)
            #########################OSU Model#####################################
            D_eddy_before_COD = g6 * np.exp(g7 * (z[LID_idx + 1:COD_idx + 1] - z_LID))
            D_eddy_after_COD = np.zeros(N - COD_idx - 1)
            D_eddy = np.concatenate((D_eddy_before_LID, D_eddy_before_COD, D_eddy_after_COD))

            
            # plt.plot(z, D_eddy)
            # plt.show()


            Delta_M = P.M_X - C.M_air

            alpha = D_X + D_eddy
            _beta = 1 / P.s_op_star * np.gradient(P.s_op_star * (D_X + D_eddy), z)
            _beta[np.isnan(_beta)] = 0
            _beta[_beta == np.inf] = 0
            _beta[_beta == -np.inf] = 0
            beta = D_X * Delta_M * C.g / C.R / P.T - P.w_air / C.year_to_sec + _beta
            _gamma = 1 / P.s_op_star * np.gradient(P.s_op_star * D_X, z)
            _gamma[np.isnan(_gamma)] = 0
            _gamma[_gamma == np.inf] = 0
            _gamma[_gamma == -np.inf] = 0
            gamma = - Delta_M * C.g / C.R / P.T * _gamma - P.lambda_X

            alpha_i_star = alpha * P.Delta_t / 2 / P.Delta_z / P.Delta_z
            beta_i_star = beta * P.Delta_t / 4 / P.Delta_z
            gamma_i_star = gamma * P.Delta_t / 2


            A = np.zeros((P.M + 1, P.M + 1))
            for i in range(P.M + 1):
                for j in range(i - 1, i + 2):
                    if j == -1 or j == P.M + 1:
                        continue
                    if j == i - 1:
                        A[i,j] = -(alpha_i_star[i] - beta_i_star[i])
                    elif j == i:
                        A[i,j] = 1 + 2 * alpha_i_star[i] - gamma_i_star[i]
                    elif j == i + 1:
                        A[i,j] = -(alpha_i_star[i] + beta_i_star[i])
            A[0,0] = 1
            A[0,1] = 0
            A[P.M,P.M-1] = -1
            A[P.M,P.M] = 1


            A_inv = np.linalg.inv(A)

            B = np.zeros((P.M + 1, P.M + 1))
            for i in range(P.M + 1):
                for j in range(i - 1, i + 2):
                    if j == -1 or j == P.M + 1:
                        continue
                    if j == i - 1:
                        B[i,j] = alpha_i_star[i] - beta_i_star[i]
                    elif j == i:
                        B[i,j] = 1 - 2 * alpha_i_star[i] + gamma_i_star[i]
                    elif j == i + 1:
                        B[i,j] = alpha_i_star[i] + beta_i_star[i]

            C_atm = 0.0 * np.ones(int(P.T / C.dt) + 1)
            C_atm[int(0.2 / C.dt) : int(0.4 / C.dt)] = 1 / (0.4 - 0.2)
            # C_atm[2] = C_atm[3] = 1
            C_i_n = C_atm[0] * np.ones((P.M + 1, 1))

            for t in tqdm(range(0, int(P.T / C.dt))):

                C_n = C_i_n[:,-1]
                C_n = C_n[:,np.newaxis]

                # print('t:', t)
                BC_n = B @ C_n
                BC_n[0] = C_atm[t + 1]
                BC_n[P.M] = 0

                C_n = A_inv @ BC_n
                C_n[C_n < 0] = 0
                C_i_n = np.hstack((C_i_n, C_n))


            C_i_n_test = C.dt * np.sum(C_i_n, axis = 1)
            print(np.max(C_i_n_test), np.argmax(C_i_n_test), np.min(C_i_n_test), np.argmin(C_i_n_test))



            
            t = np.arange(0, C.Time + C.dt, C.dt)
            t = site.sample_year - t

            # T, Z = np.meshgrid(t, z[:1000])
            # fig = plt.figure(figsize = (6,6))
            # ax = fig.add_subplot(1,1,1,projection = "3d")
            # ax.plot_surface(Z,T,C_i_n)
            # plt.show()

            x = CH4_gt[:, 0]
            y = CH4_gt[:, 1]
            CH4_gt_ = np.interp(t, x, y)
            CH4_gt_ *= C.dt
            CH4_result = C_i_n @ CH4_gt_
            CH4_result[CH4_result > 2000] = 2000

            mask = np.where((z >= start_depth) & (z <= mid_depth))

            new_RMSE = ((CH4_result[mask] - CH4_target[mask]) ** 2).mean()
            if new_RMSE < RMSE:
                print("RMSE : ", new_RMSE, "\t IMPROVED")
                D_X_best = D_X
                RMSE = new_RMSE
                axs[0].plot(z, D_X_best)
                axs[1].plot(z, CH4_result)
                plt.draw()
                plt.pause(0.3)
            else:
                print("RMSE : ", new_RMSE, "\t not IMPROVED ㅜㅜ")

            # axs[0].plot(z, D_X)
            # axs[1].plot(z, CH4_result)
            # plt.draw()
            # plt.pause(0.3)

            
            n = z_LID
            x = np.linspace(0, n, 500)
            x1, y1 = random_fourier_curve(n=n, K=8, sigma=0.3, p=2.0, x=x)
            inv_tort = np.interp(z, x1, y1)
            inv_tort[inv_tort < 0] = 0
            inv_tort[inv_tort > 1] = 1
            D_X = inv_tort * D_X_0

            count += 1

        np.savetxt("E:\\LICP\\code\\data\\SNUmodel\\icecores\\CPSW\\tortuosity.txt", D_X_best)
        np.savetxt("E:\\LICP\\code\\data\\SNUmodel\\icecores\\CPSW\\z.txt", z)
            
        















        