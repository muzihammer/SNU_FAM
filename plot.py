import numpy as np
import matplotlib.pyplot as plt


ld = np.loadtxt("E:\\LICP\\code\\data\\SNUmodel\\GAD\\DE08.txt")
cpsw = np.loadtxt("E:\\LICP\\code\\data\\SNUmodel\\GAD\\CPSW.txt")



plt.plot(- (ld[:,0] - ld[:, 0][0]), ld[:,1], c='r', label="DE-08\nCOD=72.0m")
plt.plot(- (cpsw[:,0] - cpsw[:, 0][0]), cpsw[:,1], c='b', label="CPSW\nCOD=70.0m")

plt.ylabel("Gas Age Distribution [yr-1]")
plt.legend()
plt.xlabel("Age from drilled year [yr]")
plt.ylim(0, 0.25)
plt.xlim(0, 50)
plt.show()