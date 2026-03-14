import numpy as np
import matplotlib.pyplot as plt

raw = np.loadtxt("SNUmodel\\data\\icecores\\CPSW\\density_smoothed.txt")
x = raw[:, 0]
y = raw[:, 1]
plt.scatter(x, y, marker='.')
# x_fix = np.array([x[0], x[-2]])
# y_fix = np.array([y[0], 0.9])

# model = np.poly1d(constrained_polyfit(x[:-1], y[:-1], 4, x_fix, y_fix))

# # model = np.poly1d(np.polyfit(x[:], y[:], 6))

# line = np.linspace(0, x[-2], 100)

# smoothed = model(line)

# plt.plot(line, smoothed, c='r')
plt.show()