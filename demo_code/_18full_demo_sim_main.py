
import csv
import time
import numpy as np
from numpy import random

from models.experimental import attempt_load

from adaptive_cal import adaptive_cal
from Geo import CameraGeo
from dist_cal import Distance
from gui import gui
from tools import *
from intrinsic.calibration import Calibration

class dis_log:
    def __init__(self,csv_name):

        self.csv_name = csv_name
        
        with open(csv_name,'w')as f1:
            csv_writer = csv.writer(f1)
            csv_writer.writerow([

            'GT','Estimation'

                                ])

    def update_log(self,fname,estimate,flag):                        
        common = fname.split('dis_')[1].split('.png')[0]
        gt = common.split(flag)[1].split('_')[0]
        with open(self.csv_name,'a')as f1:
            csv_writer = csv.writer(f1)
            csv_writer.writerow([

            gt,estimate

                                ])


class Mian:
    def __init__(self,):
        self.demo_gui = gui() 
        # self.dist = Distance()
    


    def main(self,im_list,img_type):
        demo = adaptive_cal()
        FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 

        calib = ['./intrinsic/f60.txt','./intrinsic/f_camera_i30.txt']
        save_dir = f'./results/distance/{FOLDER_NAME}/'
        make_dir(save_dir)
        demo.init_list(save_dir)

        dis_log1 = dis_log(save_dir+'_1_left.csv')
        dis_log2 = dis_log(save_dir+'_2_mid.csv')
        dis_log3 = dis_log(save_dir+'_3_right.csv')

        ##### param: Extrinsic => Translation
        # vehicle_height = 0.63379
        vehicle_height = 1.85
        # vehicle_height = 1.
        # obj_height = [vehicle_height + 2.9706762, vehicle_height + 3.04365305][1]
        # obj_height = [vehicle_height + 2.55700274, vehicle_height + 2.62995816][0]
        obj_height = [vehicle_height + 4.4][0]
        obj_height = 1.85
        # obj_height = 4.5
        
        ##### param: Extrinsic => Rotation
        # ## p,y,r -3.2907187963631257  1.1967836642392118  97.71791594937976
        yaw = 1.1967836642392118
        pitch = -3.2907187963631257
        roll = 97.71791594937976


        yaw = 50
        pitch = 0
        roll = 0

        #### #hyper param
        img_size= [[1920,1080],[1280,720]]
        # img_size= [[1920,1080],[1280,806]]
        cal = Calibration(calib,img_size)

        ##### param: Intrinsic
        cg = CameraGeo(height = obj_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = roll, intrinsic_matrix = cal.camera_matrix_f60)
        cg_v = CameraGeo(height = vehicle_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = roll, intrinsic_matrix = cal.camera_matrix_f60)
        opt = argument()

        _, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

        device = select_device(opt.device)

        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        if half:
            model.half()  # to FP16

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        for img in im_list:

            fname = img.split('/')[-1]
            mask_frame = f'{img_path[:-4]}/mask/{fname}'

            frame_raw = cv2.imread(img)            
            frame = cv2.imread(mask_frame)

            if img_type =='real':
                frame = cv2.undistort(frame, cal.camera_matrix_f60, cal.dist_coeffs__f60, None, cal.camera_matrix_f60)
                frame_raw = cv2.undistort(frame_raw, cal.camera_matrix_f60, cal.dist_coeffs__f60, None, cal.camera_matrix_f60)

            # Break the loop if we've reached the end of the video
            # Display the current frame
            t1 = time.time()
            print(fname)

            frame, angles = self.demo_gui.img_pro(frame_raw,frame,demo)
            self.demo_gui.t2 = time.time() - t1
            
            pitch_on = angles[0]
            yaw_on = angles[1]
            temp_angle = pitch_on * yaw_on

            if temp_angle < 1000000:
                cg = CameraGeo(height = obj_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)
                cg_v = CameraGeo(height = vehicle_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)

            if img_type == 'roll_sim':


                frame, dist_list= dis_main_roll(frame_raw,opt,cal,cg,model,stride,imgsz,device,cg_v,names,colors,half)
                max_id = len(dist_list)
                for id, dis in enumerate(sorted(dist_list)):
                    if id == 0:
                        dis_log1.update_log(fname,dis[1],flag='2_')
                    elif id == max_id-1:
                        dis_log3.update_log(fname,dis[1],flag='4_')
                    else:
                        dis_log2.update_log(fname,dis[1],flag='3_')
            else:
                frame = dis_main(frame_raw,opt,cal,cg,model,stride,imgsz,device,cg_v,names,colors,half)
            # frame = draw_ground(frame,cg.src_poi)
            cv2.imwrite(f'{save_dir}{fname}',frame)


if __name__=='__main__':
    if __package__ is None:
        import sys
        from os import path
        print(path.dirname( path.dirname( path.abspath(__file__) ) ))
        sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    else:
        pass
    Mian= Mian()
    img_type_list = ['sim','real','roll_sim']
    img_type = img_type_list[1] 

    ### real
    # img_path = '../dataset/distance/car/raw/'
    # img_path = '../dataset/distance/traffic_light/raw/'
    img_path = '../dataset/distance/niro/raw/'


    ### sim
    # img_path = '../dataset/2023_06_06_roll/raw/'
    # img_path = '../dataset/2023_06_11_distance_roll/roll_0.0/raw/'
    # img_path = '../dataset/2023_06_11_distance_roll/roll_10.0/raw/'
    # img_path = '../dataset/2023_06_11_distance_roll/roll_-10.0/raw/'
    # img_path = '../dataset/2023_06_11_distance_roll/roll_5.0/raw/'
    # img_path = '../dataset/2023_06_11_distance_roll/roll_-5.0/raw/'

    im_list = sorted(get_image_list(img_path))
    Mian.main(im_list,img_type)

