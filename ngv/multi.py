from email import header

import torch
from tool.torch_utils import *
from tool.utils import *
from models.py2_Yolov4_model import Yolov4
import os
from multiprocessing import Process

import cv2
import time
import copy
import argparse
import math
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from std_msgs.msg import Float32MultiArray 
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Transform


from API.tracker import Tracker
from API.drawer import Drawer
from calibration.calib import Calibration
from dist import Distance


class YOLOv4_Det:
    ### Need to change image_shape 
    def __init__(self, args, image_shape,image_shape_side):
        camera_path = ['calibration/f_camera_1280.txt', 'calibration/l_camera_1280.txt', 'calibration/r_camera_1280.txt']
        imu_camera_path = './calibration/camera_imu.txt'
        LiDAR_camera_path = 'calibration/f_camera_lidar_1280.txt'

        self.calib = Calibration(camera_path, imu_camera_path , LiDAR_camera_path)
        self.est_dist = Distance(self.calib)

        self.args = args
        self.img_shape = image_shape
        self.img_shape_side = image_shape_side
        self.YOLOv4 = self.LoadModel()

        self.tracker = Tracker((self.img_shape[1], self.img_shape[0]), min_hits=1, num_classes=args.n_classes, interval=args.interval)
        self.drawer = Drawer(args.namesfile)  
        self.bridge = CvBridge()

        self.cur_front_img = {'img':None, 'header':None}
        self.cur_left_img = {'img':None, 'header':None}
        self.cur_right_img = {'img':None, 'header':None}
        self.sum_front_img = {'img':None, 'header':None}
        self.sum_left_img = {'img':None, 'header':None}
        self.sum_right_img = {'img':None, 'header':None}

        self.GT_dist = None
        self.cur_pose = None

        rospy.init_node('detection1')
        rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.Front_IMGcallback)
        rospy.Subscriber('/cam0/compressed', CompressedImage, self.Left_IMGcallback)
        rospy.Subscriber('/cam1/compressed', CompressedImage, self.Right_IMGcallback)

        rospy.Subscriber('/lidar/postpoint', MarkerArray, self.LiDARcallback, queue_size=10)
        rospy.Subscriber('/Camera/Transform', Transform, self.Posecallback, queue_size=10)

        self.pub_img = rospy.Publisher('/detection/result_front', Image, queue_size=1)
        self.pub_dist = rospy.Publisher("/detection_front/dist", Float32MultiArray, queue_size=1)
        self.pub_ori_dist = rospy.Publisher("/detection_front/ori_dist", Float32MultiArray, queue_size=1)

        self.pub_2d_bbox = rospy.Publisher('/detection/bbox_2d', Float32MultiArray, queue_size=1)
        self.pub_3d_bbox = rospy.Publisher('/detection/bbox_3d', Float32MultiArray, queue_size=1)

        self.get_new_IMG_msg1 = False
        self.get_new_IMG_msg2 = False
        self.get_new_IMG_msg3 = False
        self.get_new_LiDAR_msg = False

        self.dist_arr = []
        self.ori_dist_arr = []
        self.mul_est_dist = []
        self.combine_dist = 0

    def Front_IMGcallback(self, msg):
        if not self.get_new_IMG_msg1:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            front_img = cv2.resize(front_img, (self.img_shape))

            self.cur_front_img['img'] = self.calib.undistort(front_img, 'front')
            self.cur_front_img['header'] = msg.header
            self.get_new_IMG_msg1 = True

    def Left_IMGcallback(self, msg):
        if not self.get_new_IMG_msg2:
            np_arr = np.fromstring(msg.data, np.uint8)
            left_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            left_img = cv2.resize(left_img, (self.img_shape_side))
            
            self.cur_left_img['img'] = self.calib.undistort(left_img, 'left')
            self.cur_left_img['header'] = msg.header
            self.get_new_IMG_msg2 = True

    def Right_IMGcallback(self, msg):
        if not self.get_new_IMG_msg3:
            np_arr = np.fromstring(msg.data, np.uint8)
            right_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            right_img = cv2.resize(right_img, (self.img_shape_side))
           
            self.cur_right_img['img'] = self.calib.undistort(right_img, 'right')
            self.cur_right_img['header'] = msg.header
            self.get_new_IMG_msg3 = True

    def LiDARcallback(self, msg):
        for marker in msg.markers:
            lidar_x = -marker.pose.position.x
            lidar_y = marker.pose.position.y
            lidar_z = marker.pose.position.z

        if len(msg.markers) > 0:
            # print(lidar_x, lidar_y, lidar_z)
            diag = math.sqrt(lidar_x**2+lidar_y**2)
            self.GT_dist = math.sqrt(diag**2 + lidar_z**2)
            # self.GT_dist = lidar_x

        self.get_new_LiDAR_msg = True

    def get_bbox_array_msg(self, bboxes, labels, header):
        bbox_array_msg = Float32MultiArray()
        bboxes_info = []
        for idx, bbox in enumerate(bboxes):
            left_top_x = bbox[0]
            left_top_y = bbox[1]
            right_bot_x = bbox[2]
            right_bot_y = bbox[3]
            box_info = [idx, left_top_x, left_top_y, right_bot_x, right_bot_y]
            bboxes_info.append(box_info)
        bbox_array_msg.data = sum(bboxes_info, [])
        return self.pub_2d_bbox.publish(bbox_array_msg)

    def get_3d_box_array_msg(self, pts_3d):
        pt_3d_array_msg = Float32MultiArray()
        pt_3d_info = []
        for idx, pt in enumerate(pts_3d):
            Real_X = pt[0]
            Real_Y = pt[1]
            pt_3d_info.append([Real_X, Real_Y])
        pt_3d_array_msg.data = sum(pt_3d_info, [])
        return self.pub_3d_bbox.publish(pt_3d_array_msg)

    def Posecallback(self, msg):
        self.cur_pose = [msg.rotation.x, msg.rotation.y, msg.rotation.z]
    
    def get_dist_array_msg(self, dist_arr):
        dist_array_msg = Float32MultiArray()
        for dist in dist_arr:
            dist_array_msg.data.append(dist[0])
        return dist_array_msg

    def LoadModel(self):
        model = Yolov4(n_classes=self.args.n_classes,pre_weight = args.weightfile, inference=True)
        torch.cuda.set_device(self.args.gpu_num)
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True
        print ('Current cuda device : %s'%(torch.cuda.current_device()))
        return model

    def angle_cal(self,bbox):
        for i in bbox:
            left_top_x = i[0]
            return left_top_x

    def multi_dist(self,dets_arr_left,dets_arr_right):
        ### camera fov 
        cam_fov = 120
        dist_B = 0.8 ## meter 
        dist_x1 = self.angle_cal(dets_arr_left)
        dist_x2 = self.angle_cal(dets_arr_right)
        mul_est_dist_arr = []
        ##main function
        mul_est_dist = np.abs((dist_B * self.img_shape[0] / (2* math.tan(cam_fov/2)*(dist_x1-dist_x2) ))) *10
        mul_est_dist_arr.append([round(mul_est_dist, 2)])
        return mul_est_dist_arr
    
    def cam_det(self,sum_img):
        img = cv2.resize(sum_img, (320, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = do_detect(self.YOLOv4, img, 0.5, 0.4)[0]
        return bbox

    def det_out(self,bbox):
        bbox = np.vstack(bbox)
        output = copy.copy(bbox)
        output[:,0] = (bbox[:,0] - bbox[:,2] / 2.0) * self.img_shape[0]
        output[:,1] = (bbox[:,1] - bbox[:,3] / 2.0) * self.img_shape[1]
        output[:,2] = (bbox[:,0] + bbox[:,2] / 2.0) * self.img_shape[0]
        output[:,3] = (bbox[:,1] + bbox[:,3] / 2.0) * self.img_shape[1]

        dets_arr, labels_arr = output[:,0:4], output[:,-1].astype(int)
        return dets_arr, labels_arr

    def main(self):
        try:
            new_dist_flag = False
            moving_tra, moving_det = 0., 0.
            frame_ind = 0
            ground_plane = np.array(self.calib.src_pt, np.int32)
            while not rospy.is_shutdown():
                if  self.get_new_IMG_msg1 and self.get_new_IMG_msg2 and self.get_new_IMG_msg3: #and self.get_new_IMG_msg2 and self.get_new_IMG_msg3:
                    test_start = time.time()
                    start = time.time()
                    dets_arr_front, labels_arr_front, is_dect_front = None, None, None
                    dets_arr_left, labels_arr_left, is_dect_left = None, None, None
                    dets_arr_right, labels_arr_right, is_dect_right = None, None, None

                    if np.mod(frame_ind, self.args.interval) == 0:
                        self.sum_front_img['img'] = self.cur_front_img['img']
                        self.sum_left_img['img'] = self.cur_left_img['img']
                        self.sum_right_img['img'] = self.cur_right_img['img']
                        orig_im = copy.copy(self.sum_front_img['img'])
                        orig_im = cv2.resize(orig_im, (self.img_shape))

                        # Network(YOLOv4) Input Image
                        bbox_front = self.cam_det(self.sum_front_img['img'])
                        bbox_left = self.cam_det(self.sum_left_img['img'])
                        bbox_right = self.cam_det(self.sum_right_img['img'])

                        if len(bbox_front) > 0 and len(bbox_left) > 0 and self.cur_pose is not None and len(bbox_right)>0: 
                            dets_arr_front, labels_arr_front = self.det_out(bbox_front)
                            dets_arr_left, labels_arr_left = self.det_out(bbox_left)
                            dets_arr_right, labels_arr_right = self.det_out(bbox_right)
                            
                            mul_est_dist = self.multi_dist(dets_arr_left,dets_arr_right)
                            self.mul_est_dist = mul_est_dist
                            
                            task = 'front'
                            self.ori_dist_arr,self.dist_arr, pts_3d, ground_plane, new_dist_flag = self.est_dist.getDistance(dets_arr_front, self.cur_pose,task)
                            self.pub_dist.publish(self.get_dist_array_msg(self.dist_arr))
                            self.pub_ori_dist.publish(self.get_dist_array_msg(self.ori_dist_arr))

                            weight = [0.2,0.8]
                            # self.combine_dist = [round(weight[0]*mul_est_dist[0][0] + weight[1]*self.ori_dist_arr[0][0],2)]

                            ## Compare for LiDAR
                            self.get_3d_box_array_msg(pts_3d)

                        else:
                            dets_arr_front, labels_arr_front, = np.array([]), np.array([])
                            dets_arr_left, labels_arr_left, = np.array([]), np.array([])
                            dets_arr_right, labels_arr_right = np.array([]), np.array([])
                        is_dect_front = True
                        is_dect_left = True
                        is_dect_right = True

                    elif np.mod(frame_ind, args.interval) != 0:
                        dets_arr_front, labels_arr_front = np.array([]), np.array([])
                        dets_arr_left, labels_arr_left = np.array([]), np.array([])
                        dets_arr_right, labels_arr_right = np.array([]), np.array([])
                        is_dect_front = False
                        is_dect_left = False
                        is_dect_right = False
                    pt_det = (time.time() - start)
                    ## draw standard point
                    if len(self.est_dist.center_pts) > 0:
                        for pt in self.est_dist.center_pts:
                            orig_im = cv2.line(orig_im, (int(pt[0]), int(pt[1])),(int(pt[0]), int(pt[1])), (255,255,255), 10)
                    if frame_ind != 0:
                        moving_det = (frame_ind / float(frame_ind + 1) * moving_det) + (1. / float(frame_ind + 1) * pt_det)
                    show_frame = self.drawer.draw(orig_im, dets_arr_front,labels_arr_front,self.ori_dist_arr,self.dist_arr,self.mul_est_dist,self.combine_dist, self.GT_dist, (1. / (moving_det + 1e-8)), is_tracker=False)
                    print('FPS is',(1. / (moving_det + 1e-8)))
                    if new_dist_flag:
                        show_frame = cv2.polylines(show_frame, [ground_plane], True, (0,0,255), thickness=2)
                    else:
                        show_frame = cv2.polylines(show_frame, [ground_plane], True, (0,255,0), thickness=2)
                    if len(dets_arr_front) > 0:
                        bbox_array_msg = self.get_bbox_array_msg(dets_arr_front, labels_arr_front, self.sum_front_img['header'])
                    if self.pub_img.get_num_connections() > 0:
                        msg = None
                        try:
                            msg = self.bridge.cv2_to_imgmsg(show_frame, "bgr8")
                            # msg.header = self.sum_img['header']
                            self.cur_front_img['header'] = msg.header
                        except CvBridgeError as e:
                            print(e)
                        self.pub_img.publish(msg)

                    frame_ind += 1
                    self.get_new_IMG_msg = False
                    self.get_new_IMG_msg1 = False
                    self.get_new_IMG_msg2 = False
                    self.get_new_IMG_msg3 = False
                    new_dist_flag = False
        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weightfile', default="./weight/yolov4.pth")

    parser.add_argument('--n_classes', default=80, help="Number of classes")
    parser.add_argument('--namesfile', default="data/coco.names", help="Label name of classes")
    parser.add_argument('--gpu_num', default=0, help="Use number gpu")
    parser.add_argument('--interval', default=1, help="Tracking interval")
    args = parser.parse_args()

    image_shape=(1280, 806)
    image_shape_side=(1280, 720)

    Detection = YOLOv4_Det(args, image_shape,image_shape_side)
    Detection.main()
