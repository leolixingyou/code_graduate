from matplotlib import pyplot as plt
import numpy as np

corrected = np.array([20, 13.2, 26.5, 9.8, 5.8, 23, 7, 17, 29.8, 24.3])*0.75

plt.scatter([15, 9.8, 20, 7.8, 4.9, 17.2, 5.7, 12.5, 23.5, 18.1],[20, 13.2, 26.5, 9.8, 5.8, 23, 7, 17, 29.8, 24.3],c='blue')
plt.scatter([15, 9.8, 20, 7.8, 4.9, 17.2, 5.7, 12.5, 23.5, 18.1],corrected,c='green')
plt.plot([0,30],[0,30], c='red')
plt.legend([])
plt.xlabel('LiDAR (m)')
plt.ylabel('Camera (m)')
plt.show()