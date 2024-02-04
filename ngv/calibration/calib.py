import cv2
import numpy as np
import math
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import copy

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

            # '''Mult-calibration parameters'''
            # with open('./calibration/l_camera_Rt.txt', 'r') as f:
            # # with open('./calibration/l_camera_Rtraw.txt', 'r') as f:
            #     for line in f.readlines():
            #         left_cam_rt.extend([float(i) for i in line.split(',')])

            # # self.Left_camera_RT = np.array([[left_cam_rt[0], left_cam_rt[1], left_cam_rt[2], left_cam_rt[9]],
            # #                                 [left_cam_rt[3], left_cam_rt[4], left_cam_rt[5], left_cam_rt[10]],
            # #                                 [left_cam_rt[6], left_cam_rt[7], left_cam_rt[8], left_cam_rt[11]]])
            # self.Left_camera_RT = np.array([[left_cam_rt[0], left_cam_rt[1], left_cam_rt[2]],
            #                                 [left_cam_rt[3], left_cam_rt[4], left_cam_rt[5]],
            #                                 [left_cam_rt[6], left_cam_rt[7], left_cam_rt[8]]])

            # ###left_past
            # src_pt = np.array([[5.674, -3.25],[5.674, -0.75],[10.174, -0.75],[10.174, -3.25]])
            # rst_pt = np.array([[711,468],[1020,542],[1064,438],[861,410]])
            # ###left_new
            # src_pt = np.array([[10.174, -0.75],[5.674, -0.75],[5.674, -3.25],[10.174, -3.25]])
            # rst_pt = np.array([[1095,464],[1063,559],[796,483],[923,431]])
            # left_h, status = cv2.findHomography(src_pt,rst_pt)
            # self.Left_camera_RT = left_h
            # # with open('./calibration/r_camera_Rt.txt', 'r') as f:
            # with open('./calibration/r_camera_Rt.txt', 'r') as f:
            #     for line in f.readlines():
            #         right_cam_rt.extend([float(i) for i in line.split(',')])

            # self.Right_camera_RT = np.array([[right_cam_rt[0], right_cam_rt[1], right_cam_rt[2]],
            #                                 [right_cam_rt[3], right_cam_rt[4], right_cam_rt[5]],
            #                                 [right_cam_rt[6], right_cam_rt[7], right_cam_rt[8]]])
            # #right
            # # src_pt = np.array([[6.174,1.25],[3.824,1.3],[1.824,3.0],[6.174,3.25]])#src_pt
            # # rst_pt = np.array([[34,523],[192,657],[843,643],[369,466]])#rst_pt
            # ###right_1000
            # src_pt = np.array([[10.174,0.75],[5.674,0.75],[5.674,3.25],[10.174,3.25]])
            # rst_pt = np.array([[156,446],[189,547],[492,479],[348,421]])
            # right_h, status = cv2.findHomography(src_pt,rst_pt)
            # self.Right_camera_RT = right_h

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
        self.proj_lidar2cam = np.dot(self.camera_matrix, self.lidar_RT)

        # # '''Side'''
        # self.proj_lidar2cam_le = self.Left_camera_RT
        # self.proj_lidar2cam_ri = self.Right_camera_RT

        roi_y = -3.0
        roi_x_top = 50.0
        roi_x_bottom = 4.0
        self.pt1_2 = [roi_x_top, roi_y]  # left top
        self.pt2_2 = [roi_x_top, -roi_y] # right top
        self.pt3_2 = [roi_x_bottom, -roi_y]  # right bottom 
        self.pt4_2 = [roi_x_bottom, roi_y]   # left bottom
        
        fix_z = [-1.62928751,-1.62468648, -1.68053416,-1.68513518]
        f_z = open('get_z.txt', 'w')
       
        for i, xy in enumerate([self.pt1_2, self.pt2_2, self.pt3_2, self.pt4_2]):
            xy.append(fix_z[i])
            f_z.write(str(i))
        f_z.close()

        self.pts_3d = np.array([self.pt1_2, self.pt2_2, self.pt3_2, self.pt4_2])
        pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "front")
        src_pt, self.M = self.get_M_matrix(pts_2d)
        self.src_pt = src_pt

        # left_pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "left")
        # right_pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "right")
        # l_src_pt, self.LM = self.get_M_matrix(left_pts_2d)
        # r_src_pt, self.RM = self.get_M_matrix(right_pts_2d)

        # self.l_src_pt = l_src_pt
        # self.r_src_pt = r_src_pt

    def project_3d_to_2d(self, points, task):
        num_pts = points.shape[1]
        points = np.vstack((points, np.ones((1, num_pts))))
        if task == "front":
            points = np.dot(self.proj_lidar2cam, points)
        elif task == "left":
            points = points[:2]
            num_pts_1 = points.shape[1]
            points = np.vstack((points, np.ones((1, num_pts_1))))
            points = self.proj_lidar2cam_le @ points
        elif task == "right":
            points = points[:2]
            # points = np.array([[points[0][0],points[0][1],points[0][2],points[0][3]],
            #           [points[1][0],points[1][1],points[1][2],0]])
            num_pts_1 = points.shape[1]
            points = np.vstack((points, np.ones((1, num_pts_1))))
            points = self.proj_lidar2cam_ri @ points
        points[:2, :] /= points[2, :]
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

        # M2 = cv2.getPerspectiveTransform(src_pt, self.dst_pt)
        M2,_ = cv2.findHomography(src_pt, self.dst_pt)
        
        return src_pt, M2
        
    def camera_transform(self, intrinsic, extrinsic):
        cam_R = np.array([
                        [extrinsic[0][0], extrinsic[0][1], extrinsic[0][2]],
                        [extrinsic[1][0], extrinsic[1][1], extrinsic[1][2]],
                        [extrinsic[2][0], extrinsic[2][1], extrinsic[2][2]],
                        ])

        cam_T = np.array([
                        [extrinsic[0][3]],
                        [extrinsic[1][3]],
                        [extrinsic[2][3]]
                        ])
        
        lidar_RT = self.lidar_RT
        lidar_R = np.array([
                            [lidar_RT[0][0], lidar_RT[0][1], lidar_RT[0][2]],
                            [lidar_RT[1][0], lidar_RT[1][1], lidar_RT[1][2]],
                            [lidar_RT[2][0], lidar_RT[1][2], lidar_RT[2][2]]
                            ])

        lidar_T = np.array([
                             [lidar_RT[0][3]], 
                             [lidar_RT[1][3]],
                             [lidar_RT[2][3]]
                             ])
        # ## Frist step is "R"? or "t"
        new_R = np.dot(cam_R,lidar_R)
        new_T = np.dot(cam_R,lidar_T) + cam_T
        extrinsic =np.array([
                            [new_R[0][0], new_R[0][1], new_R[0][2],new_T[0]],
                            [new_R[1][0], new_R[1][1], new_R[1][2],new_T[1]],
                            [new_R[2][0], new_R[1][2], new_R[2][2],new_T[2]]
                            ])
        return np.dot(intrinsic, extrinsic)

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


def rotationMatrixToEulerAngles(R):
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

if __name__ == "__main__":
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    camera_path = ['f_camera_1280.txt', 'l_camera_1280.txt', 'r_camera_1280.txt']
    imu_camera_path = './camera_imu.txt'
    LiDAR_camera_path = 'f_camera_lidar_1280.txt'

    calibration = Calibration(camera_path, imu_camera_path , LiDAR_camera_path)

    print(calibration.lidar_RT[:,:3])
    r = calibration.lidar_RT[:,:3]

    # Example usage:
    # R = np.array([[0.1, -0.98, 0.1], [0.6, 0.4, -0.7], [0.8, 0.3, 0.5]])
    euler_angles = rotationMatrixToEulerAngles(r)
    print('Euler angles:')
    print('Roll:', euler_angles[0])
    print('Pitch:', euler_angles[1])
    print('Yaw:', euler_angles[2])

    # ori_img = cv2.imread('test.png')
    # # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    # ori_img = calibration.undistort(ori_img)
    # cv2.imwrite('after.png', ori_img)
    # # topview_img = calibration.topview(ori_img)
    # topview_img = cv2.rotate(topview_img, cv2.ROTATE_180)
    # plt.ylim(-10, 460)
    # plt.xlim(-10, 60)

    # draw_plane = np.array(calibration.src_pt, np.int32)
    # show_frame = cv2.polylines(ori_img, [draw_plane], True, (0,255,0), thickness=3)
    # ax1.imshow(ori_img)

    # ax2.imshow(topview_img)
    # plt.show()
    # cv2.waitKey(0)
   
