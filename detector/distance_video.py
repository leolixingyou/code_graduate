
import os
import cv2
import copy
import torch
import argparse
import numpy as np

from detect_use import detect_img
from intrinsic.calibration import Calibration


MAT_EXT = ['.png']

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

def get_mat_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in MAT_EXT:
                image_names.append(apath)
    return image_names

def vector_from_vp_to_cp(vp, cp):
    return cp - vp

class CameraGeo(object):
    def __init__(self, height=1.3, yaw_deg=0, pitch_deg=-5, roll_deg=0, image_width=1024, image_height=512, field_of_view_deg=60,intrinsic_matrix=None):
        # scalar constants
        self.height = height
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.yaw_deg = yaw_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg
        # camera intriniscs and extrinsics
        self.intrinsic_matrix = intrinsic_matrix
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                            [cp*sr, cp*cr, sp],
                                            [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
        self.rotation_cam_to_road = rotation_road_to_cam.T # for rotation matrices, taking the transpose is the same as inversion
        self.translation_cam_to_road = np.array([0,-self.height,0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3,0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3,3] = self.translation_cam_to_road
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])

    def argument(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='./weights/epitone_7x_2.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='./', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/test_hsv', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        return parser.parse_args()

    def camframe_to_roadframe(self,vec_in_cam_frame):
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_camframe(self,u,v):
        # NOTE: The results depend very much on the pitch angle (0.5 degree error yields bad result)
        # Here is a paper on vehicle pitch estimation:
        # https://refubium.fu-berlin.de/handle/fub188/26792
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height*Kinv_uv_hom/denominator
    
    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y]) # read book section on coordinate systems to understand this

    def precompute_grid(self,dist=60):
        cut_v = int(self.compute_minimum_v(dist=dist)+1)
        xy = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                X,Y,Z= self.uv_to_roadXYZ_roadframe_iso8855(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):
        """
        Find cut_v such that pixels with v<cut_v are irrelevant for polynomial fitting.
        Everything that is further than `dist` along the road is considered irrelevant.
        """        
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v

def main(file,cali_path):

    ##### param: Extrinsic => Translation
    vehicle_height = 0.63379
    obj_height = [vehicle_height + 2.9706762, vehicle_height + 3.04365305][1]
    obj_height = [vehicle_height + 2.55700274, vehicle_height + 2.62995816][0]
    obj_height = 4.5
    
    ##### param: Extrinsic => Rotation
    # ## p,y,r -3.2907187963631257  1.1967836642392118  97.71791594937976
    yaw = 1.1967836642392118
    pitch = -3.2907187963631257
    roll = 97.71791594937976

    img_size= [[1,1],[1,1]]
    cal = Calibration(cali_path,img_size)

    ##### param: Intrinsic
    cg = CameraGeo(height = obj_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = roll, intrinsic_matrix = cal.f60_intrinsic)
    opt = cg.argument()

    print(file)
    fname = file.split('/')[-1][:-4]
    img = cv2.imread(file)
    img = cv2.undistort(img, cal.camera_matrix_f60, cal.dist_coeffs__f60, None, cal.camera_matrix_f60)
    boxes,im0 = detect_img(img,opt,cg)

    distance_list = []
    ### box: [xyxy,lable]
    for box in boxes:
        # print(box[1])
        if box[1].split()[0]== 'green4_h':
            principal_p_box = [int(round((box[0][0]+box[0][2])/2,0)),int(round((box[0][1]+box[0][3])/2,0))]
            distance = cg.uv_to_roadXYZ_roadframe_iso8855(principal_p_box[0],principal_p_box[1])
            print(distance)

if __name__=='__main__':

    if __package__ is None:
        import sys
        from os import path
        print(path.dirname( path.dirname( path.abspath(__file__) ) ))
        sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    else:
        pass
    flag_list = ['show','save']
    flag = flag_list[0]
    file_list = get_mat_list('../dataset/distance/')
    file_list = ['../dataset/distance/scene-2-gmsl_camera_dev_video0_compressed.png']
    print(file_list)
    cali_path = ['./intrinsic/f60.txt']

    main(file_list,cali_path)