import math
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray 
# from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform

class ROS:
    def __init__(self, calibration):
        self.calib = calibration

        self.bridge = CvBridge()
        rospy.init_node('main')
        rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.IMGcallback)

        # multi
        # rospy.Subscriber('/cam0/compressed', CompressedImage, self.Left_IMGcallback)
        # rospy.Subscriber('/cam1/compressed', CompressedImage, self.Right_IMGcallback)

        rospy.Subscriber('/vectornav/IMU', Imu, self.IMUcallback, queue_size=10)
        rospy.Subscriber('/Camera/Transform', Transform, self.Posecallback, queue_size=10)
       
        self.pub_pose = rospy.Publisher('/Camera/Transform', Transform, queue_size=1)
        # self.pub_opti_res = rospy.Publisher('/Camera/Opticalflow', Image, queue_size=1)
 
        self.cur_img = {'img':None, 'header':None}
        self.left_cur_img = {'img':None, 'header':None}
        self.right_cur_img = {'img':None, 'header':None}

        self.cur_imu = None
        self.cur_pose  = None
        self.cur_LiDAR = None

        self.get_new_IMG_msg = False
        self.get_new_Right_IMG_msg = False
        self.get_new_Left_IMG_msg = False
        self.get_new_pose_msg = False
        self.get_new_LiDAR_msg = False
        self.get_new_imu_msg = False
        
        self.bboxes = []
        self.pose_flag = True

    ### Subscriber
    def IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1280, 806))
        
        self.cur_img['img'] = self.calib.undistort(img, 'front')
        self.cur_img['header'] = msg.header
        self.get_new_IMG_msg = True

    def Left_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (self.img_shape))
        
        self.left_cur_img['img'] = self.calib.undistort(img, 'left')
        self.left_cur_img['header'] = msg.header
        self.get_new_Left_IMG_msg = True

    def Right_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (self.img_shape))
        
        self.right_cur_img['img'] = self.calib.undistort(img, 'right')
        self.right_cur_img['header'] = msg.header
        self.get_new_Right_IMG_msg = True

    def IMUcallback(self, msg):
        ## quat to euler
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
       
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(3.141592 / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.cur_imu = [0, pitch, 0]
#####
        if self.pose_flag and self.cur_imu[1] != 0:
            self.init_pose = self.cur_imu
            self.pose_flag = False

        self.cur_imu = [roll-self.init_pose[0], pitch-self.init_pose[1], 0]
# #####
        self.get_new_imu_msg = True
        
    
    def Posecallback(self, msg):
        self.cur_pose  = [msg.rotation.x, msg.rotation.y, msg.rotation.z]
        ## radian
        self.Euler = [self.cur_pose[0], self.cur_pose[1], self.cur_pose[2]] ## roll pitch yaw
        
        # self.Correct.translation_Z = imu_sensor_height
        # self.Correct.init_translation_z = msg.orientation.w - self.Correct.init_translation_z

        self.get_new_pose_msg = True

    ### Publisher
    def pose2ROS(self, cur_euler):
        pub_pose_msg = Transform()
        # est_RT_msg.translation.x = self.cur_t[0]
        # est_RT_msg.translation.y = self.cur_t[1]
        # est_RT_msg.translation.z = self.cur_t[2]
      
        pub_pose_msg.rotation.x = cur_euler[0]
        pub_pose_msg.rotation.y = cur_euler[1]
        pub_pose_msg.rotation.z = cur_euler[2]
        pub_pose_msg.rotation.w = 0

        return self.pub_pose.publish(pub_pose_msg)

    def img2ROS(self, img):
        msg = None
        try:
            msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            msg.header = self.cur_img['header']
        except CvBridgeError as e:
            print(e)

        self.pub_opti_res.publish(msg)
       