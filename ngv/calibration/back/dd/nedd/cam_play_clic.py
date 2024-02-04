#!/usr/bin/env python
import os
import rospy
from sensor_msgs.msg import CompressedImage,Image
from sensor_msgs.msg import PointCloud2
import cv2
import numpy as np
from cv_bridge import CvBridge

front_60 = 0
front_190 = 0
right = 0
left = 0

flag = False
g_u = 0
g_v = 0

bridge = CvBridge()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_txt(u,v):
    output_path = '/media/cvlab/Data1/Xingyou/NGV/Hyundai_Project/calibration/multi/'
    mkdir(output_path)
    txt_name = 'new.txt'
    # txt_name = 'left.txt'
    # txt_name = 'front_le.txt'
    # txt_name = 'right.txt'
    # txt_name = 'front_ri.txt'
    with open(output_path + txt_name, 'a') as f:
        f.write(str(u) + ',' + str(v) + '\n')


def undistort(img, camera_matrix, dist_coeffs):
    DIM = (img.shape[1], img.shape[0])
    h,w = img.shape[:2]
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), camera_matrix, DIM, cv2.CV_16SC2)
    # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # undistorted_img = img.copy()
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    return undistorted_img

def mouse_callback(event, u, v, flags, param):
    global g_u, g_v
    g_u = u
    g_v = v
    if event == cv2.EVENT_LBUTTONDOWN:
        print("u :", u, ", v :", v)
        write_txt(u,v)
 

def read_txt(path, name ):
    cam_param = []
    with open(path + name + '.txt', 'r') as f:
        f = f.read()
        f = f.replace('[','').replace(']','').replace('\n',',')
        for i in f.split(','):
            if i != '':
                cam_param.append(float(i))
    return cam_param
def msgCallback0(msg):
    img_np_arr = np.fromstring(msg.data, np.uint8)
    global left 
    left = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    # left = cv2.resize(left, (640, 403))
   
def msgCallback1(msg):
    img_np_arr = np.fromstring(msg.data, np.uint8)
    global right 
    right = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    # right = cv2.resize(right, (640, 403))

def msgCallback2(msg):
    global flag
    # if flag == True:
    # img_np_arr = np.fromstring(msg.data, np.uint8)
    global front_60 
    # front_190 = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    front_60 = bridge.imgmsg_to_cv2(msg, "bgr8")
    front_60 = cv2.resize(front_60,(1280,806))

    # front_60 =undistort(front_60, camera_matrix, dist_coeffs)
    front_60 = cv2.undistort(front_60, camera_matrix, dist_coeffs, None, camera_matrix)
    front_60 = cv2.line(front_60, (640, 0), (640, 806), (255,0,0), thickness=1)
    cv2.putText(front_60,"("+str(g_u)+","+str(g_v)+")",(g_u, g_v),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),4,lineType=cv2.LINE_AA)


    # front_60 = cv2.resize(front_60, (640, 403))
    cv2.namedWindow("front_60", 0)
    cv2.setMouseCallback("front_60", mouse_callback)
    cv2.imshow("front_60", front_60)
    k = cv2.waitKey(1)
    if k == 27:
        exit()
def msgCallback3(msg):
    global flag
    # if flag == True:
    img_np_arr = np.fromstring(msg.data, np.uint8)
    global front_190 
    front_190 = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    # front_190 = bridge.imgmsg_to_cv2(msg, "bgr8")
    front_190 = cv2.resize(front_190,(1280,806))

    # front_190 =undistort(front_190, camera_matrix, dist_coeffs)
    front_190 = cv2.undistort(front_190, camera_matrix, dist_coeffs, None, camera_matrix)
    front_190 = cv2.line(front_190, (640, 0), (640, 806), (255,0,0), thickness=1)
    cv2.putText(front_190,"("+str(g_u)+","+str(g_v)+")",(g_u, g_v),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),4,lineType=cv2.LINE_AA)


    # front_190 = cv2.resize(front_190, (640, 403))
    cv2.namedWindow("front_190", 0)
    cv2.setMouseCallback("front_190", mouse_callback)
    cv2.imshow("front_190", front_190)
    k = cv2.waitKey(1)
    if k == 27:
        exit()


def lidarCallback(msg):
    global flag
    flag = True

rospy.init_node('Image_node')
rospy.loginfo("-------subscriber_node start!-------")

# #WI30
# rospy.Subscriber("/cam0/Raw", CompressedImage, msgCallback0)
# rospy.Subscriber("/Camera1/compressed", CompressedImage, msgCallback1)
rospy.Subscriber("/gmsl_camera/port_0/cam_0/image_raw/compressed", CompressedImage, msgCallback3)
# rospy.Subscriber("/cam0/Raw", Image, msgCallback2)
# rospy.Subscriber("/cam1/Raw", Image, msgCallback2)


path = '/media/cvlab/Data1/Xingyou/NGV/good_performance/WI30/mat_inter/'
front_60 = 'WI30_1280_front_1118'
left = 'left_mat_inter'
right = 'right_mat_inter'

name = front_60

cam_param = read_txt(path, name)
camera_matrix = np.array([[cam_param[0], cam_param[1], cam_param[2]], 
                          [cam_param[3], cam_param[4], cam_param[5]], 
                          [cam_param[6], cam_param[7], cam_param[8]]])
dist_coeffs = np.array([[cam_param[9], cam_param[10], cam_param[11], cam_param[12], cam_param[13]]])

rospy.spin()
# while not rospy.is_shutdown():    
    # cv2.namedWindow('left',0)
    # cv2.imshow("left", left)
    # cv2.namedWindow('right',0)
    # cv2.imshow("right", right)
    # cv2.namedWindow('front_60',0)
    # cv2.imshow("front_60", front_60)
