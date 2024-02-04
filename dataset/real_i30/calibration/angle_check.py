
import numpy as np
import rospy
import sensor_msgs.point_cloud2
from std_msgs.msg import Float32MultiArray 
from sensor_msgs.msg import PointCloud2
from sklearn import linear_model, datasets


def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def euler2degrees( cur_pose):
    roll = math.degrees(cur_pose[0])
    pitch = math.degrees(cur_pose[1])
    yaw = math.degrees(cur_pose[2])

    return roll, pitch, yaw

def main(LiDAR_cam_path):
    lidar_calib_param = []
    with open(LiDAR_cam_path, 'r') as f:
        for line in f.readlines():
            lidar_calib_param.extend([float(i) for i in line.split(',')])

    lidar_RT = np.array([[lidar_calib_param[0], lidar_calib_param[1], lidar_calib_param[2], lidar_calib_param[9]],
                            [lidar_calib_param[3], lidar_calib_param[4], lidar_calib_param[5], lidar_calib_param[10]],
                            [lidar_calib_param[6], lidar_calib_param[7], lidar_calib_param[8], lidar_calib_param[11]]])
    print('lidar_RT')
    print(lidar_RT)
    Roatation = lidar_RT[:,:3]
    print(Roatation)
    ro = rotationMatrixToEulerAngles(Roatation)
    x1,y1,z1 = euler2degrees(ro)
    print(x1,y1,z1)
    
if __name__ == "__main__":
    path = '/media/cvlab/Data1/Xingyou/NGV/Hyundai_Project_test/calibration/'

    path_wi30 = './WI30/Hyendai/'
    
    paths = glob.glob('./data/WI30/*.png')
    paths.sort()
    print(paths)
    
    RT_path = '/media/cvlab/Data1/Xingyou/NGV/Hyundai_Project_test/calibration/'
    
    mode = []
    # mode = 'front'
    if mode == 'front':
        print('front')
        name = 'f_camera_1280'
        RT_name = 'f_camera_lidar_1280.txt'
        LiDAR_cam_path = RT_path + RT_name
        camera_matrix, dist_coeffs = read_in(path,name)
        validation(LiDAR_cam_path, camera_matrix, dist_coeffs)
