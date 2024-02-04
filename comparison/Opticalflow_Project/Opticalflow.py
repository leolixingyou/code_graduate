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

from scipy.spatial.transform import Rotation as sci_R


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

class Opticlaflow:
    def __init__(self):
        self.K = np.array([[1108.5520500997836, 0.00000000e+00, 640.0], 
                        [0.00000000e+00, 1108.5520500997836, 403.0], 
                        [0.0, 0.0, 1.0]])

        self.D = np.array([0, 0, 0, 0, 0])
        
        self.cur_img = {'img':None, 'header':None}
        self.get_new_img_msg = False
        self.preproc_ct = 0
        self.Imu_Data = None
        self.Init_IMU_Data = None
        self.Init_CAM_Data = None

        self.cur_R, self.cur_t = None, None
        self.pre_est_euler = np.array([0, 0, 0])

        self.scale = Scale()

        ### Error
        self.imu_error = []
        self.cam_error = []
        self.x_error = []
        self.y_error = []
        self.z_error = []
        self.error_ct = 1
        ###

        self.bridge = CvBridge()
    
        rospy.init_node('opticalflow')
        rospy.Subscriber('/vds_node_localhost_2211/image_raw/compressed', CompressedImage, self.SM_IMGcallback)
        # rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.SM_IMGcallback)
        rospy.Subscriber('/imu_sensor/pose', Imu, self.SM_IMUcallback, queue_size=10)

        self.pub_od = rospy.Publisher('/od_result', Image, queue_size=1)

        ## For Plot
        self.GT_rpy = rospy.Publisher('/gt_euler', Float64MultiArray, queue_size=1)
        self.Est_rpy = rospy.Publisher('/est_euler', Float64MultiArray, queue_size=1)

    def undistort(self, img):
        w,h = (img.shape[1], img.shape[0])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w,h), 0)
        result_img = cv2.undistort(img, self.K, self.D, None, newcameramtx)
        return result_img

    def SM_IMGcallback(self, msg):
        if not self.get_new_img_msg:
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.cur_img['img'] = self.undistort(img)
            self.cur_img['header'] = msg.header
            self.get_new_img_msg = True
    
    def SM_IMUcallback(self, msg):
        self.cur_imu = msg.orientation.x, msg.orientation.y, msg.orientation.z ## Euler angle (radian)
        self.Position = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z ## absolute Translation (m)

        ## R_z
        mat_yaw = np.array([[math.cos(self.cur_imu[2]), -math.sin(self.cur_imu[2]), 0],
                            [math.sin(self.cur_imu[2]), math.cos(self.cur_imu[2]), 0],
                            [0, 0, 1]])
        ## R_y
        mat_pitch = np.array([[math.cos(self.cur_imu[1]),0,math.sin(self.cur_imu[1])],
                            [0,1,0],
                            [-math.sin(self.cur_imu[1]),0,math.cos(self.cur_imu[1])]])
        ## R_x
        mat_roll = np.array([[1, 0, 0],
                            [0, math.cos(self.cur_imu[0]), -math.sin(self.cur_imu[0])],
                            [0, math.sin(self.cur_imu[0]), math.cos(self.cur_imu[0])]])

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
        _, R, t, mask = cv2.recoverPose(E, cur_kp, pre_kp, focal=self.K[0][0], pp=(self.K[0][2], self.K[1][2]))
        return R, t

    def posemsg2ROS(self, est_euler):
        est_pose_msg = Imu()
        est_pose_msg.orientation.x = est_euler[0]
        est_pose_msg.orientation.y = est_euler[1]
        est_pose_msg.orientation.z = est_euler[2]
        est_pose_msg.angular_velocity.x = self.cur_t[0][0]
        est_pose_msg.angular_velocity.y = self.cur_t[1][0]
        est_pose_msg.angular_velocity.z = self.cur_t[2][0]
        return est_pose_msg

    def getCameraRtMsg(self, position):
        camera_pose = Float64MultiArray()
        camera_pose.data = position
        return camera_pose

    def test_error(self, imu, cam):
        error_len_parm = 10

        x_error = imu[0]-cam[0]
        y_error = imu[1]-cam[1]
        z_error = imu[2]-cam[2]

        x_scale, y_scale, z_scale = 0,0,0

        for x,y,z in zip(self.x_error, self.y_error, self.z_error):
            x_scale += math.sqrt(x**2)
            y_scale += math.sqrt(y**2)
            z_scale += math.sqrt(z**2)

        
        if len(self.x_error) == error_len_parm and len(self.y_error) == error_len_parm:
            self.x_error.pop(0)
            self.y_error.pop(0)
            self.z_error.pop(0)

        self.x_error.append(x_error)
        self.y_error.append(y_error)
        self.z_error.append(z_error)

        return x_scale/len(self.x_error), y_scale/len(self.y_error), z_scale/len(self.z_error)


    def getEulerAngle(self, R, t, state, pre_kp, cur_kp):
        self.cur_R = R.dot(self.cur_R)
            
        IMU_R_x = math.atan2(self.IMU_R[2,1] , self.IMU_R[2,2])
        IMU_R_y = math.atan2(self.IMU_R[2,0], math.sqrt(self.IMU_R[0,0] * self.IMU_R[0,0] + self.IMU_R[1,0] * self.IMU_R[1,0]))
        IMU_R_z = math.atan2(self.IMU_R[1,0], self.IMU_R[0,0])

        cur_R_x = math.atan2(self.cur_R[2,1] , self.cur_R[2,2])
        cur_R_y = math.atan2(self.cur_R[2,0], math.sqrt(self.cur_R[0,0] * self.cur_R[0,0] + self.cur_R[1,0] * self.cur_R[1,0]))
        cur_R_z = math.atan2(self.cur_R[1,0], self.cur_R[0,0])

        cur_euler = Float64MultiArray()
        # cur_euler.data = [math.degrees(cur_R_y), math.degrees(cur_R_x), math.degrees(cur_R_z)]
        cur_euler.data = [(cur_R_y), (cur_R_x), (cur_R_z)]

        gt_euler = Float64MultiArray()
        # gt_euler.data = [math.degrees(IMU_R_x), math.degrees(IMU_R_y), math.degrees(IMU_R_z)]
        gt_euler.data = [(IMU_R_x), (IMU_R_y), (IMU_R_z)]

        self.Est_rpy.publish(cur_euler)
        self.GT_rpy.publish(gt_euler)

        if self.error_ct % 10 == 0:
            self.x_error = []
            self.y_error = []
            self.z_error = []

        self.x_error.append(math.sqrt((IMU_R_x-cur_R_x)**2))
        self.y_error.append(math.sqrt((IMU_R_y-cur_R_y)**2))
        self.z_error.append(math.sqrt((IMU_R_z-cur_R_z)**2))

        self.error_ct += 1

        cur_R_x, cur_R_y, cur_R_z = self.cur_R[0], self.cur_R[1], self.cur_R[2]

       
        if state > 0.25:
            x, y, z = cur_R_x, cur_R_y, cur_R_z
            self.pre_est_euler = x, y, z
            return np.array([x, y, z])
      
        else:
            return np.array([self.pre_est_euler[0], self.pre_est_euler[1], self.pre_est_euler[2]])

    def main(self):
        try:
            frame_ct,moving_fps = 0.0, 0.0

            while not rospy.is_shutdown():
                if self.get_new_img_msg and self.preproc_ct <= 1:
                    if self.preproc_ct == 0 :
                        first_frame = cv2.cvtColor(self.cur_img['img'], cv2.COLOR_BGR2GRAY) 
                        img_mask = np.zeros_like(self.cur_img['img'])

                    else :
                        second_frame = cv2.cvtColor(self.cur_img['img'], cv2.COLOR_BGR2GRAY)
                        pre_kp = self.featureDetection(first_frame, second_frame)
                        last_frame = second_frame
                        self.Init_CAM_Data = self.Imu_Data

                    self.preproc_ct += 1
                    self.get_new_img_msg = False
                
                start_fps = time.time()

                if self.get_new_img_msg and self.preproc_ct > 1:# and frame_ct % 30 == 0:
                    new_frame = cv2.cvtColor(self.cur_img['img'], cv2.COLOR_BGR2GRAY)     
                    pre_kp, cur_kp = self.featureTracking(last_frame, new_frame, pre_kp)

                    state = np.mean(np.abs(cur_kp - pre_kp)) ## detect vehicle movement
                
                    ## Update feature (At least number of point 6)
                    if  state < 0.25 or pre_kp.shape[0] < 6 or frame_ct % 2 == 0:
                        pre_kp = self.featureDetection(last_frame, new_frame)
                        pre_kp, cur_kp = self.featureTracking(last_frame, new_frame, pre_kp)
                        img_mask = np.zeros_like(self.cur_img['img'])

                    R, t = self.getRTMatrix(pre_kp, cur_kp)
                    est_euler = self.getEulerAngle(R, t, state, pre_kp, cur_kp)

                    ### Draw
                    for i,(new,old) in enumerate(zip(cur_kp, pre_kp)):
                        a,b = new.ravel()
                        c,d = old.ravel()
                        img_mask = cv2.line(img_mask, (int(a),int(b)),(int(c),int(d)), (0,255,0), 2)
                        frame = cv2.circle(self.cur_img['img'],(int(a),int(b)),3,(0,255,0),-1)
                    
                    if self.pub_od.get_num_connections() > 0:
                        msg = None
                        try:
                            msg = self.bridge.cv2_to_imgmsg(cv2.add(frame, img_mask), "bgr8")
                            msg.header = self.cur_img['header']
                        except CvBridgeError as e:
                            print(e)
                        self.pub_od.publish(msg)

                    pre_kp = cur_kp
                    last_frame = new_frame

                    end_fps = time.time()-start_fps
            
                    if frame_ct != 0:
                        moving_fps = (frame_ct / float(frame_ct + 1) * moving_fps) + (1. / float(frame_ct + 1) * end_fps)
                        print("FPS : {%0.2f}" %(1./moving_fps))

                    self.get_new_img_msg = False
                    frame_ct += 1

                
        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")

if __name__ == "__main__":
    optical = Opticlaflow()
    optical.main()
    rospy.spin()

    from matplotlib import pyplot as plt

    imu_error = optical.imu_error
    cam_error = optical.cam_error

    a_sub_b = [x for x in imu_error if x not in cam_error]
    print(a_sub_b)
