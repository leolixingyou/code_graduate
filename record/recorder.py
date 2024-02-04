import importlib
import cv2
import rospy
import numpy as np
import subprocess
import os
import copy
from calibration.calib import Calibration

def cmd_output_reader(cmd):
    cmd = cmd.split(' ')
    output = subprocess.check_output(cmd)
    output = output.decode('utf-8')
    output = output.split('\n')
    output.pop()

    return output

def topic_selecter():
    topic_list = cmd_output_reader('rostopic list')

    for i, value in enumerate(topic_list):
        print('\033[1m{0}:\033[0m {1}'.format(i, value))

    topic_index = input('\n\033[1m' +
        'Select the index of topic you want to recorde. : ' +
        '\033[0m')

    selected_topic = topic_list[int(topic_index)]

    print('\033[1mselected topic :\033[0m {0}'.format(selected_topic))

    selected_topic_type = cmd_output_reader('rostopic type {0}'.format(selected_topic))[0]

    return selected_topic, selected_topic_type

def output_path(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    file_ext = '.mp4'
    if not os.path.exists('{}'.format(dir_path) + '/results'):
       os.mkdir('{}'.format(dir_path) + '/results')

    output_path='{}'.format(dir_path) + '/results/%s%s' %(filename,file_ext)
    uniq=1
    while os.path.exists(output_path):
        output_path = '{}'.format(dir_path) + '/results/%s_%d%s' % (filename,uniq,file_ext) 
        uniq+=1
    return output_path


def recorder(width, height, filename='video'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_video_path = output_path(filename)
    return cv2.VideoWriter(result_video_path, fourcc, 30.0, (int(width), int(height)))


def viewer(img):
    global recording_toggle
    cv2.putText(img,'recording : ', (1050, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 0),2,lineType=cv2.LINE_AA)
    if recording_toggle:
        cv2.putText(img, 'on', (1200, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 200),2,lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, 'off', (1200, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 0),2,lineType=cv2.LINE_AA)
    return img
    
def ros_subscriber(topic_name, topic_type):
    topic_type = topic_type.split('/')
    module_name, class_name = '{}.msg'.format('.'.join(topic_type[:-1])), topic_type[-1]
    module = importlib.import_module(module_name)
    topic_class = getattr(module, class_name)
    rospy.Subscriber(topic_name, topic_class, callback)

def callback(msg):
    global get_new_IMG_msg, cur_img, calib

    if not get_new_IMG_msg:
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # img = cv2.resize(img, (1280, 806))
        # img = cv2.resize(img, (1280, 720))
        # img = cv2.resize(img, (1920, 1080))
        # img = cv2.resize(img, (1920, 1080))

        # cur_img['img'] = calib.undistort(img, 'front')
        cur_img['img'] = img
        cur_img['header'] = msg.header
        get_new_IMG_msg = True
        
def main():
    global calib, get_new_IMG_msg, cur_img, recording_toggle, recording_init_toggle
    # width, height = 1280, 806
    width, height = 1920, 1080
    init_img = np.ones((width, height,3),dtype=np.uint8)
    get_new_IMG_msg = False
    recording_toggle = False
    recording_init_toggle = False
 
    cur_img = {'img':init_img, 'header':None}

    camera_path = ['calibration/f_camera_1280.txt', 'calibration/l_camera_1280.txt', 'calibration/r_camera_1280.txt']
    LiDAR_camera_path = 'calibration/f_camera_lidar_1280.txt'
    calib =  Calibration(camera_path, LiDAR_camera_path)

    topic_name, topic_type = topic_selecter()

    rospy.init_node('img_recorder')
    ros_subscriber(topic_name, topic_type)

    while not rospy.is_shutdown():
        cv2.namedWindow("viewer")
        img = copy.copy(cur_img['img'])
        img = viewer(img)
        cv2.imshow('viewer', img)
        if recording_toggle and recording_init_toggle:
            video = recorder(width, height)
            recording_init_toggle = False
        if recording_toggle:
            video.write(cur_img['img'])
        get_new_IMG_msg = False
        try:
            key = cv2.waitKey(30)
            if key == 27:
                break
            if key == 114:
                if recording_toggle:
                    recording_toggle = False
                    recording_init_toggle = False
                    video.release()
                else:
                    recording_toggle = True
                    recording_init_toggle = True
        except:
            pass

if __name__ == "__main__":
    main()
