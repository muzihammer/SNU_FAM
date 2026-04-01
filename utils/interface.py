import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import utils.constants as C
from utils.site import Site
from utils.gas import Gas
from utils.profiles import Profile

class Interface:
    def __init__(self):
        df_sites = pd.read_excel(C.ROOT+"sites.xlsx")
        self.sites = list()
        for i, site in df_sites.iterrows():
            self.sites.append(Site(site["Site"], 
                                   site["T[C]"], 
                                   site["A[m ieq]"], 
                                   site["p[hPa]"], 
                                   site["H[m]"], 
                                   site["Z[m]"], 
                                   site["Year"], 
                                   site["rho_0"], 
                                   site["use_HL"] == "O"))

        df_gases = pd.read_excel(C.ROOT+"gases.xlsx")
        self.gases = list()
        for i, gas in df_gases.iterrows():
            self.gases.append(Gas(gas["Gas"], 
                                  gas["Mass [kg/mol]"], 
                                  gas["Gamma"],
                                  gas["Lambda"]))

    def run(self):
        query_1 = "원하는 지역을 선택하라냥 =^._.^= ∫\n"
        for i, data in enumerate(self.sites):
            query_1 += str(i + 1) + ". " + data.name + "\n"
        query_1 += ":\t"
        id_1 = int(input(query_1)) - 1
        site = self.sites[id_1]

        query_2 = "기준 기체를 선택하라냥 =^._.^= ∫\n"
        for i, data in enumerate(self.gases):
            query_2 += str(i + 1) + ". " + data.name + "\n"
        query_2 += ":\t"
        id_2 = int(input(query_2)) - 1
        gas = self.gases[id_2]



        z = np.arange(0, site.Z, C.dz)
        t = np.arange(0, C.Time + C.dt, C.dt)
        t = site.sample_year - t
        
        P = Profile(z, site, gas)

        s, s_cl, s_op, rho = P.porosity()
        w_air, w_ice, p = P.velocity()
        tau_DZ, tau_LIZ = P.tortuosity()
        D_X, D_eddy, D_total, s_op_star = P.diffusion()
        # gad_LID, gad_COD, gad = P.gas_age_distribution()


        fig, axs = plt.subplots(1, 9, figsize=(16, 8), sharey=True)
        fig.suptitle("Profile of Site " + site.name , fontsize=16, y=0.95)
        fig.text(0.2, 0, 'x_air = '+str(round(P.x_air * 1E3, 3)) + 'ml/kg', ha='center', va='bottom', fontsize=14)
        # fig.text(0.4, 0, ' = '+str(round(P.x_air * 1E3, 3)) + 'ml/kg', ha='center', va='bottom', fontsize=14)
        # fig.text(0.6, 0, 'x_air = '+str(round(P.x_air * 1E3, 3)) + 'ml/kg', ha='center', va='bottom', fontsize=14)
        # axs = axs.flatten()
        lid = P.z_LID
        cod = P.z_COD
        for i, ax in enumerate(axs):
            ax.axhline(lid, color='black', linestyle='-.', linewidth=1)
            ax.axhline(cod, color='black', linestyle='-.', linewidth=1)

        axs[0].plot(s, z, label="s", c='b')
        axs[0].plot(s_cl, z, label="s_cl", c='r')
        axs[0].plot(s_op, z, label="s_op", c='g')
        axs[0].plot(s_op_star, z, label="s_op_*", c='m', linestyle='--')
        axs[0].invert_yaxis()
        axs[0].text(np.mean(axs[0].get_xlim()), lid, 'LID', ha='center', va='bottom')
        axs[0].text(np.mean(axs[0].get_xlim()), cod, 'COD', ha='center', va='bottom')
        axs[0].set_ylabel("Depth (m)")
        axs[0].set_xlabel("Porosity (m3/m3)")
        axs[0].legend()
        axs[0].set_ylim(100, 0)

        axs[1].plot(rho, z, label="rho", c='b')
        axs[1].set_xlabel("Density (g/cm3)")
        axs[1].legend()


        axs[2].plot(w_ice, z, label='w_ice', c='b')
        axs[2].plot(w_air, z, label='w_air', c='r')
        axs[2].set_xlabel("Velocity (m/yr)")
        axs[2].legend()

        
        axs[3].plot(p, z, label='p/p0', c='b')
        axs[3].set_xlabel("Pressure (Pa/Pa)")
        axs[3].set_xlim(1, 2.5)
        axs[3].legend()


        axs[4].plot(D_X, z, label='D_X', c='b')
        axs[4].plot(D_eddy, z, label='D_eddy', c='r')
        axs[4].set_xlabel("Diffusion Coeff. (m2/s)")
        axs[4].legend()

        axs[5].plot(D_total, z, label='D_total', c='b')
        axs[5].set_xscale('log', base=10)
        axs[5].set_xlabel("Total Diffusion (m2/s)")
        axs[5].legend()
        axs[5].set_xlim(1E-9, 1E-4)

        axs[6].plot(D_eddy/D_total, z, label='D_eddy/D_total', c='b')
        axs[6].set_xlabel("Diffusion Ratio (m2s-1/m2s-1)")
        axs[6].legend()

        
        axs[7].plot(tau_DZ, z, label='Tortuosity_DZ', c='b')
        axs[7].set_xlabel("Tortuosity")
        axs[7].legend()

        
        axs[8].plot(tau_LIZ, z, label='Tortuosity_LIZ', c='b')
        axs[8].set_xlabel("Tortuosity")
        axs[8].legend()


        plt.show()


        P.plotGAD(cod - 10)
        P.plotGAD(63)
        P.plotGAD(cod)
        P.plotGAD(78)
        P.plotGAD(cod + 10)



        # plt.show()


        
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))


        lid = P.z_LID
        cod = P.z_COD
        for i, ax in enumerate(axs):
            ax.axvline(lid, color='black', linestyle='-.', linewidth=1)
            ax.axvline(cod, color='black', linestyle='-.', linewidth=1)

        axs[0].plot(z, D_X, label='D_X', c='b')
        axs[0].plot(z, D_eddy, label='D_eddy', c='r')
        axs[0].set_ylabel("Diffusion Coeff. (m2/s)")
        axs[0].legend()
        axs[0].set_xlim(0, 75)
        axs[0].set_ylim(0, 1.9E-5)

        axs[1].plot(z, D_total, label='D_total', c='b')
        axs[1].set_yscale('log', base=10)
        axs[1].set_ylabel("Total Diffusion (m2/s)")
        axs[1].legend()
        axs[1].set_xlim(0, 75)
        axs[1].set_ylim(1E-9, 1E-4)

        axs[2].plot(z, D_eddy/D_total, label='D_eddy/D_total', c='b')
        axs[2].set_ylabel("Diffusion Ratio (m2s-1/m2s-1)")
        axs[2].legend()
        axs[2].set_xlim(0, 75)
        axs[2].set_ylim(bottom=0)
        # axs[0].set_ylim(0, 1.9E-5)

        plt.show()

        # with open('E:\\LICP\\code\\SNUmodel\\results\\'+site.name+'.pkl', 'wb') as f:
        #     pickle.dump(P, f, protocol=pickle.HIGHEST_PROTOCOL)
