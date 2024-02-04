
import cv2
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

class Mian:
    def __init__(self,):
        self.demo_gui = gui() 
        # self.dist = Distance()
        self.u,self.v = 0,0

    def main(self,im_list):
        demo = adaptive_cal()

        FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 
        calib = ['./intrinsic/f60.txt','./intrinsic/f_camera_i30.txt']
        save_dir = f'./results/distance/{FOLDER_NAME}/'
        make_dir(save_dir)
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

        # yaw = -0.6810110838532153
        # pitch = -0.21769690844034348
        # roll = 0.0
      
        #### #hyper param
        img_size= [[1920,1080],[1280,720]]
        # img_size= [[1920,1080],[1280,806]]
        cal = Calibration(calib,img_size)

        ##### param: Intrinsic
        cg = CameraGeo(height = obj_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = roll, intrinsic_matrix = cal.camera_matrix_f60)
        cg_v = CameraGeo(height = vehicle_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = roll, intrinsic_matrix = cal.camera_matrix_f60)
        opt = cg.argument()

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

        im_list = ['../dataset/distance/car/raw/101.png']
        for img in im_list:
            global g_u, g_v

            while True:
                if yaw > 360:
                    yaw -= 360
                if yaw < -360:
                    yaw += 360
                if pitch > 360:
                    pitch -= 360
                if pitch < -360:
                    pitch += 360
                if roll > 360:
                    roll -= 360
                if roll < -360:
                    roll += 360                    

                fname = img.split('/')[-1]
                mask_frame = f'{img_path[:-4]}/mask/{fname}'

                frame_raw = cv2.imread(img)            
                frame = cv2.imread(mask_frame)            
                # Break the loop if we've reached the end of the video
                # Display the current frame
                t1 = time.time()
                
                frame, angles = self.demo_gui.img_pro(frame_raw,frame,demo)
                self.demo_gui.t2 = time.time() - t1
                temp_angle = angles[0] * angles[1]
                if temp_angle < 1000000:
                    cg = CameraGeo(height = obj_height, yaw_deg = math.degrees(angles[1]), pitch_deg =  math.degrees(angles[0]), roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)
                    cg_v = CameraGeo(height = vehicle_height, yaw_deg =  math.degrees(angles[1]), pitch_deg =  math.degrees(angles[0]), roll_deg = 0, intrinsic_matrix = cal.camera_matrix_f60)

                frame_d = dis_main(frame_raw,opt,cal,cg,model,stride,imgsz,device,cg_v,names,colors,half)
                frame_f = frame_d.copy()

                print(f'yaw is {yaw}, pitch is {pitch}, roll is {roll}, ')
                cg = CameraGeo(height = obj_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = roll, intrinsic_matrix = cal.camera_matrix_f60)
                frame = draw_ground(frame_f,cg.src_poi)
                
                key = cv2.waitKey(0) 
                key = key & 0xFF  # 添加这一行以确保键的值在0-255之间
                
                cv2.imshow(f'{save_dir}{fname}',frame)
                        
                if key == ord('q'):
                    break
                # 如果按下 'u' 键，增加亮度
                elif key == ord('u'):
                    yaw  += 10
                elif key == ord('i'):
                    yaw  -= 10
                elif key == ord('j'):
                    yaw  += 1
                elif key == ord('k'):
                    yaw  -= 1

                elif key == ord('o'):
                    pitch += 10
                elif key == ord('p'):
                    pitch -= 10
                elif key == ord('l'):
                    pitch += 1
                elif key == ord(';'):
                    pitch -= 1

                elif key == ord('t'):
                    roll += 10
                elif key == ord('y'):
                    roll -= 10
                elif key == ord('g'):
                    roll += 1
                elif key == ord('h'):
                    roll -= 1       
                elif key == ord('r'):
                    # yaw, roll, pitch = 0,0,0
                    yaw = -0.6810110838532153
                    pitch = -0.21769690844034348
                    roll = 0.0
            break
        cv2.destroyAllWindows()

if __name__=='__main__':
    if __package__ is None:
        import sys
        from os import path
        print(path.dirname( path.dirname( path.abspath(__file__) ) ))
        sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    else:
        pass
    Mian= Mian()
    img_path = '../dataset/distance/car/raw/'
    # img_path = './dataset/distance/traffic_light/raw/'
    # img_path = '../dataset/2023_06_06_roll/raw/'

    im_list = sorted(get_image_list(img_path))
    Mian.main(im_list)
    
