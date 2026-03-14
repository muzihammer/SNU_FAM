import numpy as np
import matplotlib.pyplot as plt

z = np.arange(0, 150, 0.1)

rho = np.zeros(z.shape[0])
rho[z <= 18] = (9.5513 * z[z <= 18] + 399.75) / 1000
rho[(z > 18) & (z <= 76)] = (-0.040762 * z[(z > 18) & (z <= 76)] * z[(z > 18) & (z <= 76)] + 8.8754 * z[(z > 18) & (z <= 76)] + 423.97) / 1000
rho76 = (-0.040762 * (76 * 76) + 8.8754 * 76 + 423.97) / 1000
slope76 = 8.8754 + 2 * (-0.040762 * 76)
mask3 = z > 76
rho_ice = 0.9165 - (-49 - 273.15) * 1.4438E-4 - (-49 - 273.15) ** 2 * 1.5175E-7
rho[mask3] = rho_ice - (rho_ice - rho76) * np.exp(-(z[mask3] - 76) * slope76 * 0.001 / (rho_ice - rho76))


plt.plot(z, rho)

for i in range(z.shape[0]):
    print(round(z[i], 1), rho[i])

plt.show()