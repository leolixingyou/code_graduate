import cv2
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
# from test_cali import Calibration, Plane3D
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


if __name__ == "__main__":
    # calibration = Calibration('./camera.txt', './camera_lidar.txt')
    
    # ground = np.array([[17.738, -3.5271, -1.4529], 
    #                     [21.697, -3.9168, -1.3806],
    #                     [11.723, -3.3326, -1.5452],
    #                     [11.544, -0.015451, -1.574],
    #                     [6.5706, -3.2894, -1.7737],
    #                     [17.529, 3.4781, -1.6041],
    #                     [20.615, 1.6153, -1.4931],
    #                     [8.3946, 3.3406, -1.7405],
    #                     [11.409, 0.2648, -1.556]])

    ground = np.array([[21.047, -0.3736, -1.387],
                            [22.612, 2.090, -1.387],
                            [23.885, -1.7588, -1.387],
                            [16.730, 1.4395, -1.5487],
                            [16.004, -1.2716, -1.487],
                            [16.175, -2.5947, -1.487],
                            [12.827, 2.4882, -1.587],
                            [12.902, -3.4785, -1.587],
                            [11.987, -1.2445, -1.587],
                            [8.2376, -2.4286, -1.687],
                            [8.2085, 1.5940, -1.687],
                            [8.5676, -2.4698, -1.687],
                            [5.4809, -0.4660, -1.687],
                            [5.3492, 1.4221, -1.687],
                            [5.6559, -1.9657, -1.6876]])

    pt1_2 = [47.0, 5.0]  # left top
    pt2_2 = [47.0, -5.0] # right top
    pt3_2 = [5.0, -2]  # right bottom 
    pt4_2 = [5.0, 2]   # left bottom

    XY = ground[:,:2]
    Z  = ground[:,2]
  
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),residual_threshold=0.01)
    ransac.fit(XY, Z)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    X = ground[:,0]
    Y = ground[:,1]
    Z = ground[:,2]

    predict_x = []
    predict_y = []
    predict_z = []
    
    for i in [pt1_2, pt2_2, pt3_2, pt4_2]:
        i.append(ransac.predict(np.array([[i[0], i[1]]])))
        predict_x.append(i[0])
        predict_y.append(i[1])
        predict_z.append(i[2])
    
    
    ori_vertices = [list(zip(predict_x, predict_y, predict_z))]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    print(X[outlier_mask])    
    ax.scatter(X[inlier_mask], Y[inlier_mask], Z[inlier_mask], c='b', marker='o', label='Inlier Data')
    ax.scatter(X[outlier_mask], Y[outlier_mask], Z[outlier_mask], c='r', marker='s', label='Outlier Data')
    ax.scatter(predict_x, predict_y, predict_z, c='black', marker='s', label='Custom Data')

    ori_poly = Poly3DCollection(ori_vertices, alpha=0.1, color='g')
    ax.add_collection3d(ori_poly)


    # imu_pitch_line, = self.canvas.axex2.plot(self.x, self.y, '.-', markersize=5, animated=True,  color='red', lw=1)
    # canvas.axex2.legend(loc='upper right', fontsize=7, ncol=3, shadow=True, borderaxespad=-1.5)

    ax.set_xlim(0,50)
    ax.set_xlabel('X[m]')
    ax.set_ylim(-6,6)
    ax.set_ylabel('Y[m]')
    ax.set_zlim(-3,3) 
    ax.set_zlabel('Z[m]')
    
    plt.legend()
    plt.show()

