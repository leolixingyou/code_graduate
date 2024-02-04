
import csv
import time
import torch
import numpy as np

from tools import *

from LaneDetector import LaneDetector

from NN_model import ErrorPredictionNN_5 as ErrorPredictionNN
from kalman import KalmanFilter


class adaptive_cal:
    def __init__(self):
        # self.width,self.height = (1920,1080)
        self.width,self.height = (1024,512)
        self.video = None
        self.t0 = 0
        self.count = 0
        self.current_time = 0


        
        self.model = ErrorPredictionNN()
        self.model.load_state_dict(torch.load("./weights/layers_5_pitch_yaw_model_pr.pth"))
        self.model.eval()

        # self.pitch_f, self.yaw_f = read_data()
        model_path = "./weights/fastai_model.pth"
        self.ld = LaneDetector(model_path=model_path)
        self.kf = KalmanFilter()

    def kal_pro_p(self,u_i,v_i,x):
        x_pred = self.kf.predict(x)
        x = self.kf.update(x_pred, [u_i,v_i])
        u_i_k, v_i_k = x
        return u_i_k,v_i_k,x

    def kal_pro(self,pitch,yaw,x,flag):
        x_pred = self.kf.predict(x)
        x = self.kf.update(x_pred, [pitch,yaw])
        pitch_k, yaw_k = x
        if flag == 'pp':
            if not -15<pitch_k<15 or not -15<yaw_k<15:
                pitch_k, yaw_k = self.pitch_NN_bk, self.yaw_NN_bk
        return pitch_k,yaw_k,x

    ### input is img & intrinsic
    def cal_angle(self,img,intrinsic_matricx):
        ###Param
        pitch_NN_bk, yaw_NN_bk, u_i_k, v_i_k = None, None, None, None

        width, height = img.shape[1],img.shape[0]
        try:
            u_i, v_i = vp_find(img,self.ld)

            Paula_start = time.time()
            pitch_pau_ori, yaw_pau_ori = on_the_fly(u_i,v_i,intrinsic_matricx)
            print(f'Paula_ori is {time.time()-Paula_start}')

            Lee_start = time.time()
            pitch_lee_ori, yaw_lee_ori = [np.rad2deg(x) for x in get_py_from_vp(u_i, v_i, intrinsic_matricx)]
            print(f'Lee_ori is {time.time()-Lee_start}')

            Paula_start = time.time()
            pitch_pau_ok, yaw_pau_ok, self.a = self.kal_pro(pitch_pau_ori,yaw_pau_ori,self.a,'Pau')
            print(f'Paula_ok is {time.time()-Paula_start}')
            Lee_start = time.time()
            pitch_lee_ok, yaw_lee_ok, self.b = self.kal_pro(pitch_lee_ori,yaw_lee_ori,self.b,'Lee')
            print(f'Lee_ok is {time.time()-Lee_start}')

            u_i_k, v_i_k, self.x =self.kal_pro_p(u_i,v_i,self.x)

            cp = np.array([width/2,height/2])
            vp_i = np.array([u_i,v_i])
            vp_i_k = np.array([u_i_k,v_i_k])
            
            Paula_start = time.time()
            pitch_pau_vk, yaw_pau_vk = on_the_fly(u_i_k,v_i_k,intrinsic_matricx)
            print(f'Paula_vk is {time.time()-Paula_start}')
            Lee_start = time.time()
            pitch_lee_vk, yaw_lee_vk = [np.rad2deg(x) for x in get_py_from_vp(u_i_k, v_i_k, intrinsic_matricx)]
            print(f'Lee_vk is {time.time()-Lee_start}')

            vp_i2cp = vector_from_vp_to_cp(vp_i,cp)
            vp_ik2cp = vector_from_vp_to_cp(vp_i_k,cp)

            new_vp2cp = vp_i2cp  # Replace this with the actual new data
            new_vp2cp_tensor = torch.tensor(new_vp2cp, dtype=torch.float32)

            # Prepare the input data for inference
            new_vp2cp_k = vp_ik2cp  # Replace this with the actual new data
            new_vp2cp_tensor_k = torch.tensor(new_vp2cp_k, dtype=torch.float32)
            # Perform inference
            Pro_start = time.time()
            with torch.no_grad():
                predicted_output = self.model(new_vp2cp_tensor)
                predicted_pitch_errors, predicted_yaw_errors = predicted_output[0].numpy(), predicted_output[1].numpy()
                Pro_mid = time.time()
                pitch_NN_ori, yaw_NN_ori = pitch_lee_ori + predicted_pitch_errors, yaw_lee_ori + predicted_yaw_errors
                Pro_end = time.time()
                print(f'Pro_ori is {Pro_end-Pro_start}')
                pitch_NN_ok, yaw_NN_ok, self.c = self.kal_pro(pitch_NN_ori,yaw_NN_ori,self.c,'pp')
                Pro_end2 = time.time()
                print(f'Pro_ok is {Pro_end2-Pro_end}')
                
                predicted_output_k = self.model(new_vp2cp_tensor_k)
                predicted_pitch_errors_k, predicted_yaw_errors_k = predicted_output_k[0].numpy(), predicted_output_k[1].numpy()
                pitch_NN_vk, yaw_NN_vk = pitch_lee_vk + predicted_pitch_errors_k, yaw_lee_vk + predicted_yaw_errors_k
                print(f'Pro_vk is {time.time()-Pro_end2}')

            pitch_pau_bk, yaw_pau_bk, self.d = self.kal_pro(pitch_pau_vk, yaw_pau_vk,self.d,'Pau')
            pitch_lee_bk, yaw_lee_bk, self.e = self.kal_pro(pitch_lee_vk, yaw_lee_vk,self.e,'Lee')
            pitch_NN_bk, yaw_NN_bk, self.f = self.kal_pro(pitch_NN_vk, yaw_NN_vk,self.f,'pp')

            # print('bk k is ',pitch_NN_bk, yaw_NN_bk)
            with open(self.save_dir+'phone_log.csv', "a") as f1:
                csv_writer = csv.writer(f1)
                csv_writer.writerow([
                    pitch_pau_ori, pitch_pau_vk, pitch_pau_ok, pitch_pau_bk, yaw_pau_ori, yaw_pau_vk, yaw_pau_ok, yaw_pau_bk,
                    pitch_lee_ori, pitch_lee_vk, pitch_lee_ok, pitch_lee_bk, yaw_lee_ori, yaw_lee_vk, yaw_lee_ok, yaw_lee_bk,
                    pitch_NN_ori, pitch_NN_vk, pitch_NN_ok, pitch_NN_bk, yaw_NN_ori, yaw_NN_vk, yaw_NN_ok, yaw_NN_bk
                    ])

            self.pitch_NN_b = pitch_NN_bk
            self.yaw_NN_bk = yaw_NN_bk
            # print(f'pitch_NN_bk is {pitch_NN_bk}')
            # print(f'yaw_NN_bk is {yaw_NN_bk}')
        except:
            pass
        return pitch_NN_bk, yaw_NN_bk, u_i_k, v_i_k

    def error_distribution(self,img,intrinsic_matricx):
        ###Param
        output_list = []

        width, height = img.shape[1],img.shape[0]
        try:
            u_i, v_i = vp_find(img,self.ld)

            Lee_start = time.time()
            pitch_lee_ori, yaw_lee_ori = [np.rad2deg(x) for x in get_py_from_vp(u_i, v_i, intrinsic_matricx)]

            cp = np.array([width/2,height/2])
            vp_i = np.array([u_i,v_i])
            vp_i2cp = vector_from_vp_to_cp(vp_i,cp)

            new_vp2cp = vp_i2cp  # Replace this with the actual new data
            new_vp2cp_tensor = torch.tensor(new_vp2cp, dtype=torch.float32)

            with torch.no_grad():
                predicted_output = self.model(new_vp2cp_tensor)
                predicted_pitch_errors, predicted_yaw_errors = predicted_output[0].numpy(), predicted_output[1].numpy()
                pitch_NN_ori, yaw_NN_ori = pitch_lee_ori + predicted_pitch_errors, yaw_lee_ori + predicted_yaw_errors

            output_list = [pitch_lee_ori, yaw_lee_ori,
                           pitch_NN_ori, yaw_NN_ori,
                           vp_i, vp_i2cp
                    ]
            # print(f'pitch_NN_bk is {pitch_NN_bk}')
            # print(f'yaw_NN_bk is {yaw_NN_bk}')
        except:
            pass
        return output_list

    def init_list(self,save_dir):
        self.save_dir = save_dir
        self.x, self.a, self.b, self.c, self.d, self.e, self.f = [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]
        self.yaw_NN_bk, self.pitch_NN_bk,= [], []
        with open(save_dir+'phone_log.csv', "w") as f1:
            csv_writer = csv.writer(f1)
            csv_writer.writerow([
                'paula_pitch_ori', 'paula_pitch_vk',  'paula_pitch_ok', 'paula_pitch_bk','paula_yaw_ori','paula_yaw_vk', 'paula_yaw_ok','paula_yaw_bk',
                'lee_pitch_ori', 'lee_pitch_vk',  'lee_pitch_ok', 'lee_pitch_bk','lee_yaw_ori','lee_yaw_vk', 'lee_yaw_ok','lee_yaw_bk',
                'pro_pitch_ori', 'pro_pitch_vk',  'pro_pitch_ok', 'pro_pitch_bk','pro_yaw_ori','pro_yaw_vk', 'pro_yaw_ok','pro_yaw_bk',
                'NN_pitch_ori', 'NN_pitch_vk',  'NN_pitch_ok', 'NN_pitch_bk','NN_yaw_ori','NN_yaw_vk', 'NN_yaw_ok','NN_yaw_bk'])
