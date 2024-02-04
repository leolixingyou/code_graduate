import cv2
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
# from test_cali import Calibration, Plane3D
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def read_txt():
    ground_points = []
    with open('./ground_points.txt', 'r') as f:
        for line in f.readlines():
            ground_points.append([float(i) for i in line.split(',')])
    return np.array(ground_points) 

if __name__ == "__main__":
    left_cam_rt = []
    with open('./l_camera_Rt.txt', 'r') as f:
        for line in f.readlines():
            left_cam_rt.extend([float(i) for i in line.split(',')])

    Left_camera_R = np.array([[left_cam_rt[0], left_cam_rt[1], left_cam_rt[2]],
                             [left_cam_rt[3], left_cam_rt[4], left_cam_rt[5]],
                             [left_cam_rt[6], left_cam_rt[7], left_cam_rt[8]],
                             [0,0,0,1]])
    Left_camera_T = np.array([ 
                             [1,0,0,left_cam_rt[9]],
                             [0,1,0,left_cam_rt[10]],
                             [0,0,1,left_cam_rt[11]],
                             [0,0,0,1]
                             ])
    print(Left_camera_R)
    print(Left_camera_T)
    result = np.dot(Left_camera_R, Left_camera_T)
    print(result)