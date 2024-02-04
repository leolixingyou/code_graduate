import cv2
import numpy as np
import math
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

class Plane3D:
    def __init__(self, data):
        XY = data[:,:2]
        Z  = data[:,2]
        self.ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=0.001)
        self.ransac.fit(XY, Z)
        
    def get_z(self, x, y):
        return self.ransac.predict(np.array([[x, y]]))

class Calibration:
    def __init__(self, camera_path, imu_cam_path, LiDAR_cam_path):
        if not None in [camera_path, imu_cam_path]:
            # camera parameters
            cam_param = []
            left_cam_param = []
            right_cam_param = []
            left_cam_rt = []
            right_cam_rt = []

            with open(camera_path[0], 'r') as f:
                for i in f.readlines():
                    for val in i.split(','):
                        cam_param.append(float(val))

            with open(camera_path[1], 'r') as f:
                for i in f.readlines():
                    for val in i.split(','):
                        left_cam_param.append(float(val))

            with open(camera_path[2], 'r') as f:
                for i in f.readlines():
                    for val in i.split(','):
                        right_cam_param.append(float(val))

            '''Main(Front) Camera Calibration'''
            self.camera_matrix = np.array([[cam_param[0], cam_param[1], cam_param[2]], 
                                        [cam_param[3], cam_param[4], cam_param[5]], 
                                        [cam_param[6], cam_param[7], cam_param[8]]])
            self.dist_coeffs = np.array([[cam_param[9]], [cam_param[10]], [cam_param[11]], [cam_param[12]], [cam_param[13]]])


            '''Multi-Camera parameters'''
            self.Left_camera_matrix = np.array([[left_cam_param[0], left_cam_param[1], left_cam_param[2]], 
                                        [left_cam_param[3], left_cam_param[4], left_cam_param[5]], 
                                        [left_cam_param[6], left_cam_param[7], left_cam_param[8]]])
            self.Left_dist_coeffs = np.array([[left_cam_param[9]], [left_cam_param[10]], [left_cam_param[11]], [left_cam_param[12]], [left_cam_param[13]]])

    
            self.Right_camera_matrix = np.array([[right_cam_param[0], right_cam_param[1], right_cam_param[2]], 
                                        [right_cam_param[3], right_cam_param[4], right_cam_param[5]], 
                                        [right_cam_param[6], right_cam_param[7], right_cam_param[8]]])
            self.Right_dist_coeffs = np.array([[right_cam_param[9]], [right_cam_param[10]], [right_cam_param[11]], [right_cam_param[12]], [right_cam_param[13]]])

            
            '''IMU-calibration parameters'''
            # imu_calib_param = []
            # with open(imu_cam_path, 'r') as f:
            #     for line in f.readlines():
            #         imu_calib_param.extend([float(i) for i in line.split(',')])
            # self.imu_RT = np.array([[imu_calib_param[0], imu_calib_param[1], imu_calib_param[2], imu_calib_param[9]],
            #                         [imu_calib_param[3], imu_calib_param[4], imu_calib_param[5], imu_calib_param[10]],
            #                         [imu_calib_param[6], imu_calib_param[7], imu_calib_param[8], imu_calib_param[11]]])

            '''Mult-calibration parameters'''
            with open('./calibration/l_camera_Rt.txt', 'r') as f:
                for line in f.readlines():
                    left_cam_rt.extend([float(i) for i in line.split(',')])

            self.Left_camera_RT = np.array([[left_cam_rt[0], left_cam_rt[1], left_cam_rt[2], left_cam_rt[9]],
                                            [left_cam_rt[3], left_cam_rt[4], left_cam_rt[5], left_cam_rt[10]],
                                            [left_cam_rt[6], left_cam_rt[7], left_cam_rt[8], left_cam_rt[11]]])

            with open('./calibration/r_camera_Rt.txt', 'r') as f:
                for line in f.readlines():
                    right_cam_rt.extend([float(i) for i in line.split(',')])

            self.Right_camera_RT = np.array([[right_cam_rt[0], right_cam_rt[1], right_cam_rt[2], right_cam_rt[9]],
                                            [right_cam_rt[3], right_cam_rt[4], right_cam_rt[5], right_cam_rt[10]],
                                            [right_cam_rt[6], right_cam_rt[7], right_cam_rt[8], right_cam_rt[11]]])

            '''LiDAR-calibration parameters'''
            lidar_calib_param = []
            with open(LiDAR_cam_path, 'r') as f:
                for line in f.readlines():
                    lidar_calib_param.extend([float(i) for i in line.split(',')])

            self.lidar_RT = np.array([[lidar_calib_param[0], lidar_calib_param[1], lidar_calib_param[2], lidar_calib_param[9]],
                                    [lidar_calib_param[3], lidar_calib_param[4], lidar_calib_param[5], lidar_calib_param[10]],
                                    [lidar_calib_param[6], lidar_calib_param[7], lidar_calib_param[8], lidar_calib_param[11]]])
            
        else:
            print("Check for txt files")
        
        '''Front'''
        ransac_toggle = False

        if ransac_toggle:
            self.proj_lidar2cam = np.dot(self.camera_matrix, self.lidar_RT)
        else:

            self.proj_lidar2cam = [[-8.33570514e+02, -1.66875790e+03,  4.83852076e+02],
                                   [-5.39161344e+02, -4.32031401e+01, -1.85056615e+03],
                                   [-1.29536998e+00, -1.01725090e-01,  1.00000000e+00]]

        '''Side'''
        self.proj_lidar2left_cam = self.camera_transform(self.Left_camera_matrix, self.Left_camera_RT)
        self.proj_lidar2right_cam = self.camera_transform(self.Right_camera_matrix, self.Right_camera_RT)
        
        # ground estimation using RANSAC
        ### Real
        
        self.ground = np.array([[5.0861, 2.3049, -1.683],
                               [5.4662, -2.4412, -1.687],
                                [5.3386, 0.1036, -1.687],
                                [6.2059, 2.3645, -1.670],
                                [7.3845, -3.4214, -1.728],
                                [6.4042, -0.0096, -1.6802],
                                [9.5044, -0.27613, -1.653],
                                [9.7423, 1.9523, -1.631],
                                [9.7910, -2.2472, -1.649],
                                [13.834, -3.2783, -1.6684],
                                [12.586, 2.5953, -1.6294],
                                [12.926, -0.3061, -1.639],
                                [16.415, -4.3078, -1.6852],
                                [17.579, 2.7266, -1.596],
                                [16.333, -0.0829, -1.622],
                                [22.874, -3.4997, -1.6709],
                                [22.815, -0.3841, -1.647],
                                [22.317, 3.4865, -1.633],
                                [31.057, -3.4401, -1.6627],
                                [29.655, -4.0315, -1.5925],
                                [30.299, -0.9259, -1.613],
                                [37.298, -2.7716, -1.640],
                                [37.228, 3.877, -1.641],
                                [38.504, -0.2613, -1.641]])

        roi_y = 3.0
        roi_x = 50.0
        self.pt1_2 = [roi_x, roi_y]  # left top
        self.pt2_2 = [roi_x, -roi_y] # right top
        self.pt3_2 = [4.0, -roi_y]  # right bottom 
        self.pt4_2 = [4.0, roi_y]   # left bottom


        if ransac_toggle:
            # self.ground = self.read_txt() 
            self.plane = Plane3D(self.ground)
        # Real world
            for i in [self.pt1_2, self.pt2_2, self.pt3_2, self.pt4_2]:
                i.append(self.plane.get_z(i[0], i[1]))
        else:
            # self.pt1_2 = [roi_x, roi_y]  # left top
            # self.pt2_2 = [roi_x, -roi_y] # right top
            # self.pt3_2 = [4.0, -roi_y]  # right bottom 
            # self.pt4_2 = [4.0, roi_y]
            pass
        ### 0327 change

###################### grid draw ###########################
        number_grids = 10
        max_grid = 50
        grid_size = max_grid/number_grids
        pts_leftline_3d=[]
        pts_rightline_3d=[]
        for i in range(number_grids):
            if ransac_toggle:
                pts_leftline = [i*grid_size, roi_y, self.plane.get_z(i*grid_size, roi_y)]
                pts_rightline = [i*grid_size, -roi_y, self.plane.get_z(i*grid_size, -roi_y)]
            else:
                pts_leftline = [i*grid_size, roi_y]
                pts_rightline = [i*grid_size, -roi_y]
            pts_leftline_3d.append(pts_leftline)
            pts_rightline_3d.append(pts_rightline)

        self.pts_leftline_3d = np.array(pts_leftline_3d)
        self.pts_rightline_3d = np.array(pts_rightline_3d)
        
        pts_leftline_2d = self.project_3d_to_2d(self.pts_leftline_3d.transpose(), "front")
        pts_rightline_2d = self.project_3d_to_2d(self.pts_rightline_3d.transpose(), "front")
        # print(pts_leftline_2d)
        self.src_leftline_2d = np.zeros((10,2))
        self.src_rightline_2d = np.zeros((10,2))
        for i, a in enumerate(pts_leftline_2d):
            for j, b in enumerate(a):
                self.src_leftline_2d[j][i] = np.float32(int(np.round(b)))    
        for i, a in enumerate(pts_rightline_2d):
            for j, b in enumerate(a):
                self.src_rightline_2d[j][i] = np.float32(int(np.round(b)))   
        print('################'+str(self.src_rightline_2d))


###############################################################

        ### 0327 change
        self.pts_3d = np.array([self.pt1_2, self.pt2_2, self.pt3_2, self.pt4_2])
        
        pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "front")
        left_pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "left")
        right_pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "right")

        self.pts_2d = pts_2d
        
        src_pt, self.M = self.get_M_matrix(pts_2d)
        l_src_pt, self.LM = self.get_M_matrix(left_pts_2d)
        r_src_pt, self.RM = self.get_M_matrix(right_pts_2d)

        self.src_pt = src_pt
        self.l_src_pt = l_src_pt
        self.r_src_pt = r_src_pt

        
        '''Only for Topview'''
        self.resolution = 10 # 1pixel = 10m
        self.grid_size = (int((self.pt1_2[1] - self.pt2_2[1]) * 100 / self.resolution), int((self.pt1_2[0] - self.pt3_2[0]) * 100 / self.resolution))
        
        topview_dst_pt = np.float32([[0, 0],
                                    [self.grid_size[0], 0], 
                                    [self.grid_size[0], self.grid_size[1]],
                                    [0, self.grid_size[1]]])

        self.topview_M = cv2.getPerspectiveTransform(src_pt, topview_dst_pt)
    
    def camera_transform(self, intrinsic, extrinsic):
        cam_R = np.array([[extrinsic[0][0], extrinsic[0][1], extrinsic[0][2], 0],
                            [extrinsic[1][0], extrinsic[1][1], extrinsic[1][2],0],
                            [extrinsic[2][0], extrinsic[2][1], extrinsic[2][2], 0],
                            [0, 0, 0, 1]])

        cam_t = np.array([[1, 0, 0, extrinsic[0][3]],
                            [0, 1, 0, extrinsic[1][3]],
                            [0, 0, 1, extrinsic[2][3]],
                            [0, 0, 0, 1]])

        ### Frist step is "R"? or "t"
        cam_transform_mat = np.dot(np.dot(self.lidar_RT, cam_t), cam_R)

        return np.dot(intrinsic, cam_transform_mat)

    def project_3d_to_2d(self, points, task):
        num_pts = points.shape[1]
        points = np.vstack((points, np.ones((1, num_pts))))
        print("================")
        print(points)
        if task == "front":
            points = np.dot(self.proj_lidar2cam, points)
            print(points)
        ### 0327
        # elif task == "left":
        #     points = np.dot(self.proj_lidar2left_cam, points)
        # elif task == "right":
        #     points = np.dot(self.proj_lidar2right_cam, points)
# 
        points[:2, :] /= points[2, :]*1.02
        return points[:2, :]

    def read_txt(self):
        ground_points = []
        with open('./calibration/ground_points.txt', 'r') as f:
            for line in f.readlines():
                ground_points.append([float(i) for i in line.split(',')])
        return np.array(ground_points) 

    def get_M_matrix(self, pts_2d):
        src_pt = np.float32([[int(np.round(pts_2d[0][0])), int(np.round(pts_2d[1][0]))],
                             [int(np.round(pts_2d[0][1])), int(np.round(pts_2d[1][1]))],
                             [int(np.round(pts_2d[0][2])), int(np.round(pts_2d[1][2]))],
                             [int(np.round(pts_2d[0][3])), int(np.round(pts_2d[1][3]))]])

        self.dst_pt = np.float32([[self.pt1_2[0], self.pt1_2[1]], 
                                [self.pt2_2[0], self.pt2_2[1]], 
                                [self.pt3_2[0], self.pt3_2[1]], 
                                [self.pt4_2[0], self.pt4_2[1]]])

        M2 = cv2.getPerspectiveTransform(src_pt, self.dst_pt)
        
        return src_pt, M2
        
    
    def undistort(self, img, task):
        w,h = (img.shape[1], img.shape[0])

        if task == "front":
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 0)
            result_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        elif task == "left":
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.Left_camera_matrix, self.Left_dist_coeffs, (w,h), 0)
            result_img = cv2.undistort(img, self.Left_camera_matrix, self.Left_dist_coeffs, None, newcameramtx)

        elif task == "right":
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.Right_camera_matrix, self.Right_dist_coeffs, (w,h), 0)
            result_img = cv2.undistort(img, self.Right_camera_matrix, self.Right_dist_coeffs, None, newcameramtx)

        return result_img
        
    def topview(self, img):
        import math
        topveiw_img = cv2.warpPerspective(img, self.topview_M, (self.grid_size[0], self.grid_size[1]))
        return topveiw_img
        
if __name__ == "__main__":
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    cam_p_path = "./camera.txt"
    cam_imu_cali_path = "./camera_imu_120.txt"
    cam_lidar_cali_path = "./camera_lidar_120.txt"
    calibration = Calibration(cam_p_path, cam_imu_cali_path, cam_lidar_cali_path)

    ori_img = cv2.imread('test.png')
    # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    ori_img = calibration.undistort(ori_img)
    cv2.imwrite('after.png', ori_img)
    # topview_img = calibration.topview(ori_img)
    topview_img = cv2.rotate(topview_img, cv2.ROTATE_180)
    plt.ylim(-10, 460)
    plt.xlim(-10, 60)

    draw_plane = np.array(calibration.src_pt, np.int32)
    show_frame = cv2.polylines(ori_img, [draw_plane], True, (0,255,0), thickness=3)
    ax1.imshow(ori_img)

    ax2.imshow(topview_img)
    plt.show()
    cv2.waitKey(0)
   
