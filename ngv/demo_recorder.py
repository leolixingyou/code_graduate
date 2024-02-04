import re
import numpy as np
import cv2
import os

import rospy
import copy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, Imu
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge
from queue import Empty, Queue
import pandas as pd

bridge = CvBridge()
tasks = ['left','front','right']
q_imgs, q_dist_befores, q_dist_afters = {}, {}, {}
imgs, dist_befores, dist_afters = {}, {}, {}
q_values = [q_imgs, q_dist_befores, q_dist_afters]
values = [imgs, dist_befores, dist_afters]
img_shape = [1280,806]
g_list_imu_pitch, g_list_cam_pitch = {}, {}
g_path = os.path.dirname(os.path.abspath(__file__))
recording_toggle = False
recording_toggle_init = False
video = {}

for task in tasks:
    imgs[task] = np.zeros((3,3,3),dtype=np.uint8)
    dist_befores[task] = 0
    dist_afters[task] = 00
    q_imgs[task] = Queue()
    q_dist_befores[task] = Queue()
    q_dist_afters[task] = Queue()
    video[task] = None
    g_list_imu_pitch[task] = []
    g_list_cam_pitch[task] = []

def callback_cam_left(msg):
    global q_imgs
    if q_imgs['left'].empty():
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        img = bridge.imgmsg_to_cv2(msg, "bgr8")   
        q_imgs['left'].put(img)

def callback_cam_front(msg):
    global q_imgs
    if q_imgs['front'].empty():
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    
        img = bridge.imgmsg_to_cv2(msg, "bgr8")   
        q_imgs['front'].put(img)

def callback_cam_right(msg):
    global q_imgs
    if q_imgs['right'].empty():
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    
        img = bridge.imgmsg_to_cv2(msg, "bgr8")   
        q_imgs['right'].put(img)

def front_distance_before_callback(msg):
    # print(msg.data)
    if msg.data != ():
        if q_dist_befores['front'].empty():
            distance = copy.deepcopy(msg.data[0])
            q_dist_befores['front'].put(distance)

def front_distance_after_callback(msg):
    # print(msg.data)
    if msg.data != ():
        if q_dist_afters['front'].empty():
            distance = copy.deepcopy(msg.data[0])
            q_dist_afters['front'].put(distance)

def left_distance_before_callback(msg):
    if msg.data != ():
        if q_dist_befores['left'].empty():
            distance = copy.deepcopy(msg.data[0])
            q_dist_befores['left'].put(distance)

def left_distance_after_callback(msg):
    if msg.data != ():
        if q_dist_afters['left'].empty():
            distance = copy.deepcopy(msg.data[0])
            q_dist_afters['left'].put(distance)

def right_distance_before_callback(msg):
    if msg.data != ():
        if q_dist_befores['right'].empty():
            distance = copy.deepcopy(msg.data[0])
            q_dist_befores['right'].put(distance)

def right_distance_after_callback(msg):
    if msg.data != ():
        if q_dist_afters['right'].empty():
            distance = copy.deepcopy(msg.data[0])
            q_dist_afters['right'].put(distance)

def overlay_pose_plot(img, dist_before, dist_after, task):
    global g_list_imu_pitch, g_list_cam_pitch, img_shape
    
    myX, myY, myW, myH = 0, 10, 941, 211

    imu_pitch = dist_before 
    cam_pitch = dist_after

    imu_pitch = imu_pitch/6 - 5
    cam_pitch = cam_pitch/6 - 5

    scale = 5

    g_list_imu_pitch[task].append(imu_pitch)
    g_list_cam_pitch[task].append(cam_pitch)
    df = pd.DataFrame([{'1. uncorreted dist' : g_list_imu_pitch[task], '2. corrected dst' : g_list_cam_pitch[task]}])
    if not os.path.exists('./sheng1.csv'):
        df.to_csv('./sheng1.csv', index=False, mode='w')
    else:
        df.to_csv('./sheng1.csv', index=False, mode='a', header=False)

    if len(g_list_imu_pitch[task]) > 50*scale:
        del(g_list_imu_pitch[task][0])
    if len(g_list_cam_pitch[task]) > 50*scale:
        del(g_list_cam_pitch[task][0])

    resolution_scale_factor = 20
    plot_x, plot_y = 50*resolution_scale_factor*scale, 10*resolution_scale_factor+1
    img_plot = np.zeros((plot_y,plot_x,3),dtype=np.uint8)
    img_plot[:,:,0] = 156
    img_plot[:,:,1] = 138
    img_plot[:,:,2] = 98

    for x1,cam in enumerate(g_list_cam_pitch[task]):
        if x1 == 0: continue
        x2 = x1-1
        y2 = -1*(g_list_cam_pitch[task][x2]*resolution_scale_factor) + plot_y/2
        y1 = -1*(g_list_cam_pitch[task][x1]*resolution_scale_factor) + plot_y/2
        cv2.line(img_plot, \
            (x2*resolution_scale_factor,max(min(int(y2),plot_y-1),0)), \
            (x1*resolution_scale_factor,max(min(int(y1),plot_y-1),0)), \
                (112,195,66),thickness=5,lineType=cv2.LINE_AA)

    for x1,imu in enumerate(g_list_imu_pitch[task]):
        if x1 == 0: continue
        x2 = x1-1
        y2 = -1*(g_list_imu_pitch[task][x2]*resolution_scale_factor) + plot_y/2
        y1 = -1*(g_list_imu_pitch[task][x1]*resolution_scale_factor) + plot_y/2
        cv2.line(img_plot, \
            (x2*resolution_scale_factor,max(min(int(y2),plot_y-1),0)), \
            (x1*resolution_scale_factor,max(min(int(y1),plot_y-1),0)), \
                (0, 0, 255),thickness=2,lineType=cv2.LINE_AA)
    # print(img.shape)
    img_plot_small = cv2.resize(img_plot, (myW, myH))
    # cv2.putText(img_plot,'  60',(0, 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    cv2.putText(img_plot_small,'  50 m',(0, 38),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    cv2.putText(img_plot_small,'  40 m',(0, 72),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    cv2.putText(img_plot_small,'  30 m',(0, 105),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    cv2.putText(img_plot_small,'  20 m',(0, 138),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    cv2.putText(img_plot_small,'  10 m',(0, 172),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    # cv2.putText(img_plot,'   0',(0, 200),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    # cv2.putText(img,'deg',(800, 105),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    img[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_plot_small,None)
    # cv2.putText(img,'-5 sec',(0, 200),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    # cv2.putText(img,'0 sec',(1000, 200),cv2.FONT_HERSHEY_SIMPLEX,0.6,(62, 52, 28),2,lineType=cv2.LINE_AA)
    width = 1280
    height = 806 if task == 'front' else 720
    img_resize = cv2.resize(img, (width, height))
    cv2.putText(img_resize,'recording : ', (1050, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(62, 52, 28),2,lineType=cv2.LINE_AA)
    if recording_toggle:
        cv2.putText(img_resize, 'on', (1200, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 200),2,lineType=cv2.LINE_AA)
    else:
        cv2.putText(img_resize, 'off', (1200, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(62, 52, 28),2,lineType=cv2.LINE_AA)  

    return img, img_resize

def subscriber():
    rospy.init_node('demo', anonymous=True)
    # rospy.Subscriber('/cam0/compressed', CompressedImage, callback_cam_left)
    # rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, callback_cam_front) 
    # rospy.Subscriber('/cam1/compressed', CompressedImage, callback_cam_right)

    # rospy.Subscriber('/detection_left/result_left', Image, callback_cam_left) 
    rospy.Subscriber('/detection/result2', Image, callback_cam_left)     
    # rospy.Subscriber('/detection_front/result_front', Image, callback_cam_front)
    rospy.Subscriber('/detection/result_front', Image, callback_cam_front)
    # rospy.Subscriber('/detection_right/result_right', Image, callback_cam_right)  
    rospy.Subscriber('/detection/result3', Image, callback_cam_right)

    rospy.Subscriber('/detection_front/dist', Float32MultiArray, front_distance_after_callback, queue_size=10)
    rospy.Subscriber('/detection_front/ori_dist', Float32MultiArray, front_distance_before_callback, queue_size=10)
    rospy.Subscriber('/detection_left/dist', Float32MultiArray, left_distance_after_callback, queue_size=10)
    rospy.Subscriber('/detection_left/ori_dist', Float32MultiArray, left_distance_before_callback, queue_size=10)
    rospy.Subscriber('/detection_right/dist', Float32MultiArray, right_distance_after_callback, queue_size=10)
    rospy.Subscriber('/detection_right/ori_dist', Float32MultiArray, right_distance_before_callback, queue_size=10)

def video_recording(task):
    width = 1920
    height = 1208 if task == 'front' else 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'divx')
    result_video_path = '{0}/{1}.mp4'.format(g_path,task)
    return cv2.VideoWriter(result_video_path, fourcc, 10.0, (int(width), int(height)))

if __name__=='__main__':
    subscriber()
    img_shape[0] = 1920
    img_shape[1] = 1208 if task == 'front' else 1080
    # img_shape[1] = 1080
    imgs_raw, imgs_show = {}, {}

    while not rospy.is_shutdown():
        key = -1
        if recording_toggle and recording_toggle_init:
            for task in tasks:
                video[task] = video_recording(task)
            recording_toggle_init = False
        for task in tasks:
            if not q_imgs[task].empty():
                imgs[task] = q_imgs[task].get()
            if not q_dist_befores[task].empty():
                dist_befores[task] = q_dist_befores[task].get()
            if not q_dist_afters[task].empty():
                dist_afters[task] = q_dist_afters[task].get()
            cv2.namedWindow(task)
            img_shape[1] = 1208 if task == 'front' else 1080
            imgs[task] = cv2.resize(imgs[task], img_shape)
            imgs_raw[task], imgs_show[task] = overlay_pose_plot(imgs[task], dist_befores[task], dist_afters[task], task)
            cv2.imshow(task, imgs_show[task])
            # cv2.imshow(task, imgs_raw[task])
            
            if recording_toggle:
                if task == 'front':
                    print(imgs_raw[task].shape)
                video[task].write(imgs_raw[task])
            ch = cv2.waitKey(10)
            if ch != -1 : key = ch
        if key == 27:
            for task in tasks:
                video[task].release()
            break
        if key == 114:
            if recording_toggle:
                recording_toggle = False
                for task in tasks:
                    video[task].release()
            else:
                recording_toggle = True
                recording_toggle_init = True
