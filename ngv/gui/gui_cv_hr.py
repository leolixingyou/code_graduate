from grpc import dynamic_ssl_server_credentials
import numpy as np
import cv2

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, Imu
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge

from queue import Queue

import math
import tf

g_imgBig = np.zeros((3,3,3),dtype=np.uint8)
bridge = CvBridge()

queue_img_front = Queue()
queue_img_left = Queue()
queue_img_right = Queue()

g_imu_init_pose = None
g_imu_pose_flag = True
queue_degrees_imu = Queue()
g_degree_imu = [0,0,0]
queue_degrees_cam = Queue()
g_degree_cam = [0,0,0]

queue_left_distance_gt = Queue()
queue_left_distance_before = Queue()
queue_left_distance_after = Queue()
queue_front_distance_gt = Queue()
queue_front_distance_before = Queue()
queue_front_distance_after = Queue()
queue_right_distance_gt = Queue()
queue_right_distance_before = Queue()
queue_right_distance_after = Queue()

g_list_imu_pitch = []
g_list_cam_pitch = []

def overlay_cam_front():
    myX, myY, myW, myH = 750, 255, 423, 238
    if not queue_img_front.empty():
        img_front_src = queue_img_front.get()
        img_front_small = cv2.resize(img_front_src, (myW, myH))
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_front_small,None)

def overlay_cam_left():
    myX, myY, myW, myH = 247, 255, 423, 238
    if not queue_img_left.empty():
        img_left_src = queue_img_left.get()
        img_left_small = cv2.resize(img_left_src, (myW, myH))
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_left_small,None)

def overlay_cam_right():
    myX, myY, myW, myH = 1267, 255, 423, 238
    if not queue_img_right.empty():
        img_right_src = queue_img_right.get()
        img_right_small = cv2.resize(img_right_src, (myW, myH))
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_right_small,None)

def callback_cam_left(msg):
    if queue_img_left.qsize() < 2:
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        queue_img_left.put(img)

def callback_cam_front(msg):
    if queue_img_front.qsize() < 2:
        img_front_src = bridge.imgmsg_to_cv2(msg, "bgr8")
        queue_img_front.put(img_front_src)

def callback_cam_right(msg):
    if queue_img_right.qsize() < 2:
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        queue_img_right.put(img)

# def front_distance_befor_callback(msg):
#     try:
#         distance = copy.deepcopy(round(msg.data[0],2))
#         front_distance_before = distance

#     except Exception as e:
#         print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

# def front_distance_after_callback(msg):
#     try:
#         distance = copy.deepcopy(round(msg.data[0],2))
#         front_distance_after = distance

#     except Exception as e:
#         print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

# def LiDARcallback(msg):
#     try:
#         for marker in msg.markers:
#             lidar_x = -marker.pose.position.x
#             lidar_y = marker.pose.position.y
#             lidar_z = marker.pose.position.z
        
#         if len(msg.markers) > 0:
#             GT_dist = math.sqrt(lidar_x**2+lidar_y**2)
#             self.front_distance_gt = round(GT_dist, 2)

def overlay_text_front():
    gt_X, gt_Y, before_X, before_Y, after_X, after_Y = 1040, 584, 940, 642, 940, 708
    error_before_X, error_before_Y, error_after_X, error_after_Y = 1085, 642, 1085, 708
    text1 = '98.12'
    cv2.putText(g_imgBig,text1,(gt_X,gt_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(156, 138, 98),2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,text1,(before_X, before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(156, 138, 98),2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,text1,(after_X,after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(156, 138, 98),2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,text1,(error_before_X, error_before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(156, 138, 98),2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,text1,(error_after_X, error_after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(156, 138, 98),2,lineType=cv2.LINE_AA)


def overlay_pose_plot():
    global g_list_imu_pitch, g_list_cam_pitch, g_degree_imu, g_degree_cam

    myX, myY, myW, myH = 750, 780, 941, 211
    if not queue_degrees_imu.empty():
        g_degree_imu = queue_degrees_imu.get()
    if not queue_degrees_cam.empty():
        g_degree_cam = queue_degrees_cam.get()

    imu_pitch = g_degree_imu[1]
    cam_pitch = g_degree_cam[1]

    g_list_imu_pitch.append(imu_pitch)
    g_list_cam_pitch.append(cam_pitch)
    if len(g_list_imu_pitch) > 50:
        del(g_list_imu_pitch[0])
    if len(g_list_cam_pitch) > 50:
        del(g_list_cam_pitch[0])

    resolution_scale_factor = 20
    plot_x, plot_y = 50*resolution_scale_factor, 10*resolution_scale_factor+1
    img_plot = np.zeros((plot_y,plot_x,3),dtype=np.uint8) + 255

    for x1,cam in enumerate(g_list_cam_pitch):
        if x1 == 0: continue
        x2 = x1-1
        y2 = -1*(g_list_cam_pitch[x2]*resolution_scale_factor) + plot_y/2
        y1 = -1*(g_list_cam_pitch[x1]*resolution_scale_factor) + plot_y/2
        cv2.line(img_plot, \
            (x2*resolution_scale_factor,max(min(int(y2),plot_y-1),0)), \
            (x1*resolution_scale_factor,max(min(int(y1),plot_y-1),0)), \
                (0,0,255),thickness=1,lineType=cv2.LINE_AA)

    for x1,imu in enumerate(g_list_imu_pitch):
        if x1 == 0: continue
        x2 = x1-1
        y2 = -1*(g_list_imu_pitch[x2]*resolution_scale_factor) + plot_y/2
        y1 = -1*(g_list_imu_pitch[x1]*resolution_scale_factor) + plot_y/2
        cv2.line(img_plot, \
            (x2*resolution_scale_factor,max(min(int(y2),plot_y-1),0)), \
            (x1*resolution_scale_factor,max(min(int(y1),plot_y-1),0)), \
                (255,0,0),thickness=1,lineType=cv2.LINE_AA)

    img_plot_small = cv2.resize(img_plot, (myW, myH))
    g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_plot_small,None)

def callback_imu(msg):
    global g_imu_pose_flag, g_imu_init_pose
    quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    cur_imu = [euler[0], euler[1], 0]

    if g_imu_pose_flag and cur_imu[1] != 0:
        g_imu_init_pose = cur_imu
        g_imu_pose_flag = False

    cur_imu = [euler[0]-g_imu_init_pose[0], euler[1]-g_imu_init_pose[1], 0]

    ## radian to degree
    roll = math.degrees(cur_imu[0])
    pitch = math.degrees(cur_imu[1])
    yaw = math.degrees(cur_imu[2])

    print(pitch)
    queue_degrees_imu.put([roll, pitch, yaw])

def callback_img_pose(msg):
        roll = -math.degrees(round(msg.rotation.x,2))
        pitch = -math.degrees(round(msg.rotation.y,2))
        yaw = -math.degrees(round(msg.rotation.z,2))
        queue_degrees_cam.put([roll, pitch, yaw])

if __name__=='__main__':
    #sizeBigImg = (1280,720)

    imgBG = cv2.imread('resources/bg.png')

    rospy.init_node('GUI', anonymous=True)
    rospy.Subscriber('/cam0/compressed', CompressedImage, callback_cam_left)
    rospy.Subscriber('/od_result', Image, callback_cam_front) 
    rospy.Subscriber('/cam1/compressed', CompressedImage, callback_cam_right)
    rospy.Subscriber('/vectornav/IMU', Imu, callback_imu, queue_size=1)
    rospy.Subscriber('/Camera/Transform', Transform, callback_img_pose, queue_size=10)
    # rospy.Subscriber('/detection/dist', Float32MultiArray, front_distance_after_callback, queue_size=10)
    # rospy.Subscriber('/detection/ori_dist', Float32MultiArray, front_distance_befor_callback, queue_size=10)
    # rospy.Subscriber('/lidar/postpoint', MarkerArray, LiDARcallback, queue_size=10)

    while not rospy.is_shutdown():
        g_imgBig = imgBG.copy()

        overlay_cam_front()
        overlay_cam_left()
        overlay_cam_right()

        overlay_pose_plot()

        overlay_text_front()

        cv2.namedWindow("pyngv", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("pyngv",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("pyngv", g_imgBig)
        try:
            key = cv2.waitKey(100)
            if key == 27:
                break
        except:
            pass
