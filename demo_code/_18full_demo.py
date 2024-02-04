
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

    def main(self,im_list,save_dir):
        demo = adaptive_cal()
        calib = ['./intrinsic/f60.txt','./intrinsic/f_camera_i30.txt']

        demo.init_list(save_dir)
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


        cap_raw = cv2.VideoCapture(im_list)
        cap = cv2.VideoCapture(f'{im_list[:26]}/mask/{im_list[30:]}')
        if not cap.isOpened() and not cap_raw.isOpened() :
            print('Error: Could not open video file.')
            exit()
        
        count = 0
        frame_count = 0
        dist_log = []
        img_log = []
        while True:
            # Read the next frame from the video
            try:
                ret, frame = cap.read()
                ret_raw, frame_raw = cap_raw.read()
                # Break the loop if we've reached the end of the video

                frame = cv2.undistort(frame, cal.camera_matrix_f60, cal.dist_coeffs__f60, None, cal.camera_matrix_f60)
                frame_raw = cv2.undistort(frame_raw, cal.camera_matrix_f60, cal.dist_coeffs__f60, None, cal.camera_matrix_f60)

                if not ret_raw and not ret:
                    break
                # Display the current frame
                frame, angles = self.demo_gui.img_pro_demo(frame_raw,frame,demo)
                
                pitch_on = angles[0]
                yaw_on = angles[1]
                # print(f'pitch_on is {pitch_on}')
                # print(f'yaw_on is {yaw_on}')
                temp_angle = pitch_on * yaw_on
                cg = CameraGeo(height = obj_height, yaw_deg = yaw_on, pitch_deg = pitch_on, roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)
                cg_v = CameraGeo(height = vehicle_height, yaw_deg = yaw_on, pitch_deg = pitch_on, roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)
                if count == 0 and temp_angle < 1000000:
                    cg = CameraGeo(height = obj_height, yaw_deg = yaw_on, pitch_deg = pitch_on, roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)
                    cg_v = CameraGeo(height = vehicle_height, yaw_deg = yaw_on, pitch_deg = pitch_on, roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)
                    count +=1

                if temp_angle < 1000000:
                    cg.update_angle(yaw_on,pitch_on)

                frame, dis_list= dis_main_roll(frame_raw,opt,cal,cg,model,stride,imgsz,device,cg_v,names,colors,half)
                frame, dis_result= draw_gt(frame, dis_list,frame_count)
                # frame = draw_ground(frame,cg.src_poi)
                dist_log.append(dis_result)
                img_log.append(frame)
                cv2.namedWindow('Video')
                cv2.imshow('Video', frame)
                # Press 'q' to exit the loop early
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            except:
                break
            frame_count += 1
            print(f'frame is {frame_count}')
        cap.release()
        figure_size = (18,9)
        font_size = 25
        plot_dist_performance(dist_log, figure_size, font_size, save_dir)
        save_video(img_log,save_dir)

if __name__=='__main__':
    if __package__ is None:
        import sys
        from os import path
        print(path.dirname( path.dirname( path.abspath(__file__) ) ))
        sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    else:
        pass
    mian= Mian()

    ### real
    img_path = '../dataset/distance/video/raw/'

    im_list = sorted(get_video_list(img_path))
    FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 
    for n in im_list:
        fname = n.split(os.sep)[-1][:-4]

        save_dir = f'./results/demo/temp_{FOLDER_NAME}/{fname}/'
        make_dir(save_dir)
        mian.main(n, save_dir)

