import numpy as np
import cv2
import math
import time

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Transform
from std_msgs.msg import Bool

from calibration.calib import Calibration
import ROS_tools


class Scale:
    def __init__(self):
        self.new_cloud = None
        self.old_cloud = None

    def triangulatePoints(self, K, R, t, pre_kp, cur_kp):
        # The canonical matrix (set as the origin)
        P0 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
        P0 = K.dot(P0)
        # Rotated and translated using P0 as the reference point
        P1 = np.hstack((R, t))
        P1 = K.dot(P1)
        # Reshaped the point correspondence arrays to cv2.triangulatePoints format
        point1 = pre_kp.reshape(2, -1)
        point2 = cur_kp.reshape(2, -1)

        return cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]

    def getRelativeScale(self, K, cur_R, cur_t, pre_kp, cur_kp):
        self.new_cloud = self.triangulatePoints(K, cur_R, cur_t, pre_kp, cur_kp)

        if self.last_cloud is None:
            self.last_cloud = self.new_cloud

        min_idx = min([self.new_cloud.shape[0], self.last_cloud.shape[0]])
        ratios = []  # List to obtain all the ratios of the distances
        for i in range(min_idx):
            if i > 0:
                Xk = self.new_cloud[i]
                p_Xk = self.new_cloud[i - 1]
                Xk_1 = self.last_cloud[i]
                p_Xk_1 = self.last_cloud[i - 1]

                if np.linalg.norm(p_Xk - Xk) != 0:
                    ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

        d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio

        self.last_cloud = self.new_cloud

        return d_ratio
        
    def getAbsoluteScale(self, Imu_Data, prev_IMU_Data): 
        x_curv = Imu_Data[0]
        y_curv = Imu_Data[1]
        z_curv = Imu_Data[2]

        x_prev = prev_IMU_Data[0]
        y_prev = prev_IMU_Data[1]
        z_prev = prev_IMU_Data[2]

        abs_scale = abs(np.sqrt((x_curv - x_prev) * (x_curv - x_prev) + (y_curv - y_prev) * (y_curv - y_prev) + (z_curv - z_prev) * (z_curv - z_prev)))

        prev_IMU_Data = [x_curv, y_curv, z_curv]

        return abs_scale, prev_IMU_Data


class PoseEstimation:
    def __init__(self, calibration):
        self.ROS = ROS_tools.ROS(calibration)

        self.K = calibration.camera_matrix
        self.D = calibration.dist_coeffs

        self.preproc_ct = 0

        self.Init_IMU_Data = None
        self.cur_R, self.cur_t = None, None
        self.pre_est_euler = np.array([0, 0, 0])

        self.scale = Scale()
        
    def Euler2RotationMat(self, msg):
        self.ROS.cur_imu = [self.ROS.cur_imu[0], self.ROS.cur_imu[1], self.ROS.cur_imu[2]]
     
        ## R_z
        mat_yaw = np.array([[math.cos(self.ROS.cur_imu[2]), -math.sin(self.ROS.cur_imu[2]), 0],
                            [math.sin(self.ROS.cur_imu[2]), math.cos(self.ROS.cur_imu[2]), 0],
                            [0, 0, 1]])
        ## R_y
        mat_pitch = np.array([[math.cos(self.ROS.cur_imu[1]),0,math.sin(self.ROS.cur_imu[1])],
                            [0,1,0],
                            [-math.sin(self.ROS.cur_imu[1]),0,math.cos(self.ROS.cur_imu[1])]])
        ## R_x
        mat_roll = np.array([[1, 0, 0],
                            [0, math.cos(self.ROS.cur_imu[0]), -math.sin(self.ROS.cur_imu[0])],
                            [0, math.sin(self.ROS.cur_imu[0]), math.cos(self.ROS.cur_imu[0])]])

        self.IMU_R = np.dot(mat_yaw, np.dot(mat_pitch, mat_roll))
        
    def featureTracking(self, image_ref, image_cur, ref_kp):
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = (23,23), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        cur_kp, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, ref_kp, None, **lk_params) 
        st = st.reshape(st.shape[0])
        pre_kp, cur_kp = ref_kp[st == 1], cur_kp[st == 1]
        return pre_kp, cur_kp

    def featureDetection(self, first_frame, second_frame):
        det = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        pre_kp = det.detect(first_frame)
        pre_kp = np.array([x.pt for x in pre_kp], dtype=np.float32)
        pre_kp, cur_kp = self.featureTracking(first_frame, second_frame, pre_kp)

        if self.preproc_ct == 1:
            self.cur_R, self.cur_t = self.getRTMatrix(pre_kp, cur_kp)

        return pre_kp

    def getRTMatrix(self, pre_kp, cur_kp):
        E, mask = cv2.findEssentialMat(cur_kp, pre_kp, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)  
        _, R, t, mask = cv2.recoverPose(E, cur_kp, pre_kp, focal=self.K[0][0], pp = (self.K[0][2], self.K[1][2]))
        return R, t

    def getEulerAngle(self, R, t, state, pre_kp, cur_kp):
        # absolute_scale, prev_IMU_Data = self.scale.getAbsoluteScale(self.ROS.cur_imu, self.Init_IMU_Data)
        # self.Init_IMU_Data = prev_IMU_Data
        # relative_scale = self.scale.getRelativeScale(self.K, self.cur_R, self.cur_t, pre_kp, cur_kp)
        # self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
        # self.cur_t = self.cur_t + self.cur_R.dot(t)
        self.cur_R = R.dot(self.cur_R)
        
        cur_R_x = math.atan2(self.cur_R[2,1] , self.cur_R[2,2])
        cur_R_y = math.atan2(self.cur_R[2,0], math.sqrt(self.cur_R[0,0] * self.cur_R[0,0] + self.cur_R[1,0] * self.cur_R[1,0]))
        cur_R_z = math.atan2(self.cur_R[1,0], self.cur_R[0,0])

        # cur_euler_cam = [cur_R_y, cur_R_x, cur_R_z]
        # cur_euler_cam = [self.ROS.cur_imu[0], self.ROS.cur_imu[1], self.ROS.cur_imu[2]]

        cur_euler_imu = [self.ROS.cur_imu[0], self.ROS.cur_imu[1], self.ROS.cur_imu[2]]
        print(cur_euler_imu)
        # self.ROS.pose2ROS(cur_euler_cam)
        self.ROS.pose2ROS(cur_euler_imu)
    
        if state > 0.25:
            x, y, z = cur_R_x, cur_R_y, cur_R_z
            self.pre_est_euler = x, y, z
            return np.array([x, y, z])
      
        else:
            return np.array([self.pre_est_euler[0], self.pre_est_euler[1], self.pre_est_euler[2]])

    def getCameraRtMsg(self, position):
        camera_pose = Float64MultiArray()
        camera_pose.data = position
        return camera_pose

    def main(self):
        try:
            frame_ct = 0
            while not rospy.is_shutdown():
                if self.ROS.get_new_IMG_msg and self.preproc_ct <= 1:
                    if self.preproc_ct == 0 :
                        first_frame = cv2.cvtColor(self.ROS.cur_img['img'], cv2.COLOR_BGR2GRAY) 
                        img_mask = np.zeros_like(self.ROS.cur_img['img'])
 
                    else :
                        second_frame = cv2.cvtColor(self.ROS.cur_img['img'], cv2.COLOR_BGR2GRAY)
                        pre_kp = self.featureDetection(first_frame, second_frame)
                        last_frame = second_frame
                        self.Init_IMU_Data = self.ROS.cur_imu

                    self.preproc_ct += 1
                    self.ROS.get_new_IMG_msg = False
                
                if self.ROS.get_new_IMG_msg and self.preproc_ct > 1:# and frame_ct % 30 == 0:
                    new_frame = cv2.cvtColor(self.ROS.cur_img['img'], cv2.COLOR_BGR2GRAY)     
                    pre_kp, cur_kp = self.featureTracking(last_frame, new_frame, pre_kp)

                    state = np.mean(np.abs(cur_kp - pre_kp)) ## detect vehicle movement
                    
                    if state < 0.25:
                        pub_state_msg = Bool()
                        pub_state_msg = True
                        self.ROS.pub_state.publish(pub_state_msg)

                    ## Update feature (At least number of point 6)
                    if  state < 0.25 or pre_kp.shape[0] < 6 or frame_ct % 2 == 0:
                        pre_kp = self.featureDetection(last_frame, new_frame)
                        pre_kp, cur_kp = self.featureTracking(last_frame, new_frame, pre_kp)
                        img_mask = np.zeros_like(self.ROS.cur_img['img'])

                    R, t = self.getRTMatrix(pre_kp, cur_kp)
                    est_euler = self.getEulerAngle(R, t, state, pre_kp, cur_kp)

                    ## Draw
                    for i,(new,old) in enumerate(zip(cur_kp, pre_kp)):
                        a,b = new.ravel()
                        c,d = old.ravel()
                        img_mask = cv2.line(img_mask, (int(a),int(b)),(int(c),int(d)), (0,255,0), 2)
                        frame = cv2.circle(self.ROS.cur_img['img'],(int(a),int(b)),3,(0,255,0),-1)

                    # if self.ROS.pub_opti_res.get_num_connections() > 0:
                    #     result_img = cv2.add(frame, img_mask)
                    #     self.ROS.img2ROS(result_img)

                    pre_kp = cur_kp
                    last_frame = new_frame

                self.ROS.get_new_IMG_msg = False
                frame_ct += 1
    
        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")
