import os
import cv2
import csv
import copy
import torch
import numpy as np

from solutions.lane_detection.lane_detector import LaneDetector
from solutions.camera_calibration.calibrated_lane_detector import get_py_from_vp

from real_data import *
from kalman import KalmanFilter
from _4vector_correct import *
from _6NN_model import judge_layers

class img_pro_class:
    def __init__(self,task,SAVE_DIR,weights):
        self.save_dir = SAVE_DIR
        # self.width,self.height = (1920,1080)
        self.width,self.height = (1024,512)
        self.video = None
        self.t0 = 0
        self.count = 0
        self.current_time = 0
        self.task = task
        
        num_layer = int(weights.split('_')[1])
        print(f'num_layer is {num_layer}')
        self.model = judge_layers(num_layer)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()
        self.weight = weights

        # self.pitch_f, self.yaw_f = read_data()
        model_path = "./solutions/lane_detection/fastai_model.pth"
        self.ld = LaneDetector(model_path=model_path)
        self.kf = KalmanFilter()
        self.kf_f = KalmanFilter()
        self.get_new_info=False

    def recorder(self,width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(self.result_video_path, fourcc, 30.0, (int(width), int(height)))

    def output_path(self,filename):
        dir_path = self.save_dir

        file_ext = '.mp4'
        make_dir(dir_path)
        output_path= dir_path + '/%s%s' %(filename,file_ext)
        uniq=1
        while os.path.exists(output_path):
            output_path = dir_path + '/%s_%d%s' % (filename,uniq,file_ext) 
            uniq+=1
        return output_path, dir_path

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
                pitch_k,yaw_k=self.pitch_tf[-1],self.yaw_tf[-1]
        return pitch_k,yaw_k,x

    def cal_angle(self,img,gt_pitch,gt_yaw):
        ###Param
        field_of_view_deg = 60 # degree
        cam_height = 2.0  # meter
        ###Param
        width, height = img.shape[1],img.shape[0]
        
        intrinsic_matricx = CameraGeometry(height=cam_height, image_width=width, image_height=height, field_of_view_deg=field_of_view_deg).intrinsic_matrix
        if self.task == 'video':
            pitch_gt, yaw_gt = 0,0
        else:
            pitch_gt, yaw_gt =gt_pitch,gt_yaw

        t1 = time.time()

        u_i, v_i = vp_find(img,self.ld)
        u_i_k, v_i_k, self.x =self.kal_pro_p(u_i,v_i,self.x)
        
        seg_time = time.time() -t1

        cp = np.array([width/2,height/2])
        vp_i = np.array([u_i,v_i])
        vp_i_k = np.array([u_i_k,v_i_k])
        
        vp_i2cp = vector_from_vp_to_cp(vp_i,cp)
        vp_ik2cp = vector_from_vp_to_cp(vp_i_k,cp)

        # Prepare the input data for inference
        new_vp2cp = vp_i2cp  # Replace this with the actual new data
        new_vp2cp_tensor = torch.tensor(new_vp2cp, dtype=torch.float32)

        new_vp2cp_k = vp_ik2cp  # Replace this with the actual new data
        new_vp2cp_tensor_k = torch.tensor(new_vp2cp_k, dtype=torch.float32)

        # Perform inference
        with torch.no_grad():
            t2 = time.time()
            predicted_output = self.model(new_vp2cp_tensor)
            predicted_pitch_errors, predicted_yaw_errors = predicted_output[0].numpy(), predicted_output[1].numpy()
            model_time = time.time() - t2
            predicted_output_k = self.model(new_vp2cp_tensor_k)
            predicted_pitch_errors_k, predicted_yaw_errors_k = predicted_output_k[0].numpy(), predicted_output_k[1].numpy()

        # # ### get pitch & yaw originally
        # pitch_pau_ori, yaw_pau_ori = on_the_fly(u_i,v_i,intrinsic_matricx)
        # pitch_lee_ori, yaw_lee_ori = [np.rad2deg(x) for x in get_py_from_vp(u_i, v_i, intrinsic_matricx)]
        # pitch_NN_ori, yaw_NN_ori = pitch_lee_ori + predicted_pitch_errors, yaw_lee_ori + predicted_yaw_errors,
        
        # # ### get pitch & yaw with KF with VP
        # pitch_pau_vk, yaw_pau_vk = on_the_fly(u_i_k,v_i_k,intrinsic_matricx)
        # pitch_lee_vk, yaw_lee_vk = [np.rad2deg(x) for x in get_py_from_vp(u_i_k, v_i_k, intrinsic_matricx)]
        # pitch_NN_vk, yaw_NN_vk = pitch_lee_vk + predicted_pitch_errors_k, yaw_lee_vk + predicted_yaw_errors_k

        # # ### get pitch & yaw with Post KF with ORI
        # pitch_pau_ok, yaw_pau_ok, self.a = self.kal_pro(pitch_pau_ori,yaw_pau_ori,self.a,'Pau')
        # pitch_lee_ok, yaw_lee_ok, self.b = self.kal_pro(pitch_lee_ori,yaw_lee_ori,self.b,'Lee')
        # pitch_NN_ok, yaw_NN_ok, self.c = self.kal_pro(pitch_NN_ori,yaw_NN_ori,self.c,'pp')

        # # ### get pitch & yaw with Post KF with VK-> both K (BK)
        # pitch_pau_bk, yaw_pau_bk, self.d = self.kal_pro(pitch_pau_vk, yaw_pau_vk,self.d,'Pau')
        # pitch_lee_bk, yaw_lee_bk, self.e = self.kal_pro(pitch_lee_vk, yaw_lee_vk,self.e,'Lee')
        # pitch_NN_bk, yaw_NN_bk, self.f = self.kal_pro(pitch_NN_vk, yaw_NN_vk,self.f,'pp')

        # ### Paula
        pitch_pau_ori, yaw_pau_ori = on_the_fly(u_i,v_i,intrinsic_matricx)

        pitch_pau_vk, yaw_pau_vk = on_the_fly(u_i_k,v_i_k,intrinsic_matricx)

        pitch_pau_ok, yaw_pau_ok, self.a = self.kal_pro(pitch_pau_ori,yaw_pau_ori,self.a,'Pau')

        pitch_pau_bk, yaw_pau_bk, self.d = self.kal_pro(pitch_pau_vk, yaw_pau_vk,self.d,'Pau')
        
        # ### Lee
        pitch_lee_ori, yaw_lee_ori = [np.rad2deg(x) for x in get_py_from_vp(u_i, v_i, intrinsic_matricx)]

        pitch_lee_vk, yaw_lee_vk = [np.rad2deg(x) for x in get_py_from_vp(u_i_k, v_i_k, intrinsic_matricx)]

        pitch_lee_ok, yaw_lee_ok, self.b = self.kal_pro(pitch_lee_ori,yaw_lee_ori,self.b,'Lee')

        pitch_lee_bk, yaw_lee_bk, self.e = self.kal_pro(pitch_lee_vk, yaw_lee_vk,self.e,'Lee')

        # ### Proposed
        t3 = time.time()
        pitch_NN_ori, yaw_NN_ori = pitch_lee_ori + predicted_pitch_errors, yaw_lee_ori + predicted_yaw_errors
        error_cal_time = time.time() - t3

        t3 = time.time()
        pitch_NN_vk, yaw_NN_vk = pitch_lee_vk + predicted_pitch_errors_k, yaw_lee_vk + predicted_yaw_errors_k
        error_cal_time_check = time.time() - t3

        t3 = time.time()
        pitch_NN_ok, yaw_NN_ok, self.c = self.kal_pro(pitch_NN_ori,yaw_NN_ori,self.c,'pp')
        kf_time = time.time() - t3

        t3 = time.time()
        pitch_NN_bk, yaw_NN_bk, self.f = self.kal_pro(pitch_NN_vk, yaw_NN_vk,self.f,'pp')
        kf_time_check = time.time() - t3


        print('ok k is ',pitch_NN_ok, yaw_NN_ok)
        print('bk k is ',pitch_NN_bk, yaw_NN_bk)
        self.update_list(
                    pitch_gt,yaw_gt, 
                    pitch_pau_ori, pitch_pau_vk, pitch_pau_ok, pitch_pau_bk, yaw_pau_ori, yaw_pau_vk, yaw_pau_ok, yaw_pau_bk,
                    pitch_lee_ori, pitch_lee_vk, pitch_lee_ok, pitch_lee_bk, yaw_lee_ori, yaw_lee_vk, yaw_lee_ok, yaw_lee_bk,
                    pitch_NN_ori, pitch_NN_vk, pitch_NN_ok, pitch_NN_bk, yaw_NN_ori, yaw_NN_vk, yaw_NN_ok, yaw_NN_bk
        )


        with open(self.name_2 + '.csv','a') as f2:
            csv_writer = csv.writer(f2)
            csv_writer.writerow([
                pitch_gt,yaw_gt, 
                pitch_pau_ori, pitch_pau_vk, pitch_pau_ok, pitch_pau_bk, yaw_pau_ori, yaw_pau_vk, yaw_pau_ok, yaw_pau_bk,
                pitch_lee_ori, pitch_lee_vk, pitch_lee_ok, pitch_lee_bk, yaw_lee_ori, yaw_lee_vk, yaw_lee_ok, yaw_lee_bk,
                pitch_NN_ori, pitch_NN_vk, pitch_NN_ok, pitch_NN_bk, yaw_NN_ori, yaw_NN_vk, yaw_NN_ok, yaw_NN_bk
                ])
        return [seg_time, error_cal_time, error_cal_time_check, kf_time, kf_time_check]


    def image_pro(self,image,gt_pitch,gt_yaw):
        img = cv2.imread(image)
        img = cv2.resize(img, (self.width,self.height))
        cur_img = copy.copy(img)
        return self.cal_angle(cur_img,gt_pitch,gt_yaw)
    
    def init_list(self,flag):
        self.x, self.a, self.b, self.c, self.d, self.e, self.f = [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]
        self.time_tf, self.time_img = [],[]
        self.pitch_tf, self.yaw_tf = [],[]

        self.pitch_gt, self.yaw_gt= [], []
        self.pitch_lee_ori, self.pitch_lee_vk, self.pitch_lee_ok, self.pitch_lee_bk,= [], [], [], []
        self.pitch_pau_ori, self.pitch_pau_vk, self.pitch_pau_ok, self.pitch_pau_bk,= [], [], [], []
        self.pitch_NN_ori, self.pitch_NN_vk, self.pitch_NN_ok, self.pitch_NN_bk,= [], [], [], []
        self.yaw_lee_ori, self.yaw_lee_vk, self.yaw_lee_ok, self.yaw_lee_bk,= [], [], [], []
        self.yaw_pau_ori, self.yaw_pau_vk, self.yaw_pau_ok, self.yaw_pau_bk,= [], [], [], []
        self.yaw_NN_ori, self.yaw_NN_vk, self.yaw_NN_ok, self.yaw_NN_bk,= [], [], [], []

        self.error_pitch_lee_ori, self.error_pitch_lee_vk, self.error_pitch_lee_ok, self.error_pitch_lee_bk,= [], [], [], []
        self.error_pitch_pau_ori, self.error_pitch_pau_vk, self.error_pitch_pau_ok, self.error_pitch_pau_bk,= [], [], [], []
        self.error_pitch_NN_ori, self.error_pitch_NN_vk, self.error_pitch_NN_ok, self.error_pitch_NN_bk,= [], [], [], []
        self.error_yaw_lee_ori, self.error_yaw_lee_vk, self.error_yaw_lee_ok, self.error_yaw_lee_bk,= [], [], [], []
        self.error_yaw_pau_ori, self.error_yaw_pau_vk, self.error_yaw_pau_ok, self.error_yaw_pau_bk,= [], [], [], []
        self.error_yaw_NN_ori, self.error_yaw_NN_vk, self.error_yaw_NN_ok, self.error_yaw_NN_bk,= [], [], [], []

        self.seg_time, self.error_cal_time, self.error_cal_time_check, self.kf_time, self.kf_time_check = [],[],[],[],[]


        self.name_2 = self.save_dir + '/' + f'{flag}_log_camera'

        with open(self.name_2 + '.csv', "w") as f2:
            csv_writer = csv.writer(f2)
            csv_writer.writerow([
                'gt_pitch','gt_yaw',
                'paula_pitch_ori', 'paula_pitch_vk',  'paula_pitch_ok', 'paula_pitch_bk','paula_yaw_ori','paula_yaw_vk', 'paula_yaw_ok','paula_yaw_bk',
                'lee_pitch_ori', 'lee_pitch_vk',  'lee_pitch_ok', 'lee_pitch_bk','lee_yaw_ori','lee_yaw_vk', 'lee_yaw_ok','lee_yaw_bk',
                'pro_pitch_ori', 'pro_pitch_vk',  'pro_pitch_ok', 'pro_pitch_bk','pro_yaw_ori','pro_yaw_vk', 'pro_yaw_ok','pro_yaw_bk'])

    def init_list_f(self):
        self.x, self.a, self.b, self.c, self.d, self.e, self.f = [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]
        self.time_tf, self.time_img = [],[]
        self.pitch_tf, self.yaw_tf = [],[]

        self.pitch_gt, self.yaw_gt= [], []
        self.pitch_lee_ori, self.pitch_lee_vk, self.pitch_lee_ok, self.pitch_lee_bk,= [], [], [], []
        self.pitch_pau_ori, self.pitch_pau_vk, self.pitch_pau_ok, self.pitch_pau_bk,= [], [], [], []
        self.pitch_NN_ori, self.pitch_NN_vk, self.pitch_NN_ok, self.pitch_NN_bk,= [], [], [], []
        self.yaw_lee_ori, self.yaw_lee_vk, self.yaw_lee_ok, self.yaw_lee_bk,= [], [], [], []
        self.yaw_pau_ori, self.yaw_pau_vk, self.yaw_pau_ok, self.yaw_pau_bk,= [], [], [], []
        self.yaw_NN_ori, self.yaw_NN_vk, self.yaw_NN_ok, self.yaw_NN_bk,= [], [], [], []

        self.error_pitch_lee_ori, self.error_pitch_lee_vk, self.error_pitch_lee_ok, self.error_pitch_lee_bk,= [], [], [], []
        self.error_pitch_pau_ori, self.error_pitch_pau_vk, self.error_pitch_pau_ok, self.error_pitch_pau_bk,= [], [], [], []
        self.error_pitch_NN_ori, self.error_pitch_NN_vk, self.error_pitch_NN_ok, self.error_pitch_NN_bk,= [], [], [], []
        self.error_yaw_lee_ori, self.error_yaw_lee_vk, self.error_yaw_lee_ok, self.error_yaw_lee_bk,= [], [], [], []
        self.error_yaw_pau_ori, self.error_yaw_pau_vk, self.error_yaw_pau_ok, self.error_yaw_pau_bk,= [], [], [], []
        self.error_yaw_NN_ori, self.error_yaw_NN_vk, self.error_yaw_NN_ok, self.error_yaw_NN_bk,= [], [], [], []

        self.seg_time, self.error_cal_time, self.error_cal_time_check, self.kf_time, self.kf_time_check = [],[],[],[],[]


    def init_csv_f(self,save_dir,flag):
        self.name_2 = save_dir + '/' + f'{flag}_log_camera'

        with open(self.name_2 + '.csv', "w") as f2:
            csv_writer = csv.writer(f2)
            csv_writer.writerow([
                'gt_pitch','gt_yaw',
                'paula_pitch_ori', 'paula_pitch_vk',  'paula_pitch_ok', 'paula_pitch_bk','paula_yaw_ori','paula_yaw_vk', 'paula_yaw_ok','paula_yaw_bk',
                'lee_pitch_ori', 'lee_pitch_vk',  'lee_pitch_ok', 'lee_pitch_bk','lee_yaw_ori','lee_yaw_vk', 'lee_yaw_ok','lee_yaw_bk',
                'pro_pitch_ori', 'pro_pitch_vk',  'pro_pitch_ok', 'pro_pitch_bk','pro_yaw_ori','pro_yaw_vk', 'pro_yaw_ok','pro_yaw_bk'])


    def update_list(self,
                    pitch_gt,yaw_gt, 
                    pitch_pau_ori, pitch_pau_vk, pitch_pau_ok, pitch_pau_bk, yaw_pau_ori, yaw_pau_vk, yaw_pau_ok, yaw_pau_bk,
                    pitch_lee_ori, pitch_lee_vk, pitch_lee_ok, pitch_lee_bk, yaw_lee_ori, yaw_lee_vk, yaw_lee_ok, yaw_lee_bk,
                    pitch_NN_ori, pitch_NN_vk, pitch_NN_ok, pitch_NN_bk, yaw_NN_ori, yaw_NN_vk, yaw_NN_ok, yaw_NN_bk
    ):

        self.pitch_gt.append(round(float(pitch_gt),3))
        self.yaw_gt.append(round(float(yaw_gt),3))
        self.pitch_pau_ori.append(round(float(pitch_pau_ori),3))
        self.pitch_pau_vk.append(round(float(pitch_pau_vk),3))
        self.pitch_pau_ok.append(round(float(pitch_pau_ok),3))
        self.pitch_pau_bk.append(round(float(pitch_pau_bk),3))

        self.yaw_pau_ori.append(round(float(yaw_pau_ori),3))
        self.yaw_pau_vk.append(round(float(yaw_pau_vk),3))
        self.yaw_pau_ok.append(round(float(yaw_pau_ok),3))
        self.yaw_pau_bk.append(round(float(yaw_pau_bk),3))

        self.pitch_lee_ori.append(round(float(pitch_lee_ori),3))
        self.pitch_lee_vk.append(round(float(pitch_lee_vk),3))
        self.pitch_lee_ok.append(round(float(pitch_lee_ok),3))
        self.pitch_lee_bk.append(round(float(pitch_lee_bk),3))

        self.yaw_lee_ori.append(round(float(yaw_lee_ori),3))
        self.yaw_lee_vk.append(round(float(yaw_lee_vk),3))
        self.yaw_lee_ok.append(round(float(yaw_lee_ok),3))
        self.yaw_lee_bk.append(round(float(yaw_lee_bk),3))

        self.pitch_NN_ori.append(round(float(pitch_NN_ori),3))
        self.pitch_NN_vk.append(round(float(pitch_NN_vk),3))
        self.pitch_NN_ok.append(round(float(pitch_NN_ok),3))
        self.pitch_NN_bk.append(round(float(pitch_NN_bk),3))

        self.yaw_NN_ori.append(round(float(yaw_NN_ori),3))
        self.yaw_NN_vk.append(round(float(yaw_NN_vk),3))
        self.yaw_NN_ok.append(round(float(yaw_NN_ok),3))
        self.yaw_NN_bk.append(round(float(yaw_NN_bk),3))

        self.error_pitch_pau_ori.append(round(float(pitch_gt),3)-round(float(pitch_pau_ori),3))
        self.error_pitch_pau_vk.append(round(float(pitch_gt),3)-round(float(pitch_pau_vk),3))
        self.error_pitch_pau_ok.append(round(float(pitch_gt),3)-round(float(pitch_pau_ok),3))
        self.error_pitch_pau_bk.append(round(float(pitch_gt),3)-round(float(pitch_pau_bk),3))

        self.error_yaw_pau_ori.append(round(float(yaw_gt),3) - round(float(yaw_pau_ori),3))
        self.error_yaw_pau_vk.append(round(float(yaw_gt),3) - round(float(yaw_pau_vk),3))
        self.error_yaw_pau_ok.append(round(float(yaw_gt),3) - round(float(yaw_pau_ok),3))
        self.error_yaw_pau_bk.append(round(float(yaw_gt),3) - round(float(yaw_pau_bk),3))


        self.error_pitch_lee_ori.append(round(float(pitch_gt),3)-round(float(pitch_lee_ori),3))
        self.error_pitch_lee_vk.append(round(float(pitch_gt),3)-round(float(pitch_lee_vk),3))
        self.error_pitch_lee_ok.append(round(float(pitch_gt),3)-round(float(pitch_lee_ok),3))
        self.error_pitch_lee_bk.append(round(float(pitch_gt),3)-round(float(pitch_lee_bk),3))

        self.error_yaw_lee_ori.append(round(float(yaw_gt),3) - round(float(yaw_lee_ori),3))
        self.error_yaw_lee_vk.append(round(float(yaw_gt),3) - round(float(yaw_lee_vk),3))
        self.error_yaw_lee_ok.append(round(float(yaw_gt),3) - round(float(yaw_lee_ok),3))
        self.error_yaw_lee_bk.append(round(float(yaw_gt),3) - round(float(yaw_lee_bk),3))

        self.error_pitch_NN_ori.append(round(float(pitch_gt),3)-round(float(pitch_NN_ori),3))
        self.error_pitch_NN_vk.append(round(float(pitch_gt),3)-round(float(pitch_NN_vk),3))
        self.error_pitch_NN_ok.append(round(float(pitch_gt),3)-round(float(pitch_NN_ok),3))
        self.error_pitch_NN_bk.append(round(float(pitch_gt),3)-round(float(pitch_NN_bk),3))

        self.error_yaw_NN_ori.append(round(float(yaw_gt),3) - round(float(yaw_NN_ori),3))
        self.error_yaw_NN_vk.append(round(float(yaw_gt),3) - round(float(yaw_NN_vk),3))
        self.error_yaw_NN_ok.append(round(float(yaw_gt),3) - round(float(yaw_NN_ok),3))
        self.error_yaw_NN_bk.append(round(float(yaw_gt),3) - round(float(yaw_NN_bk),3))

    def plot_curve(self,axs,metric):
        idx = list(range(len(self.pitch_gt)))
        # Set the title for each subplot
        axs[0].set_title('Angle_pitch')
        axs[1].set_title('Angle_yaw')
        axs[0].set_ylabel('Degree')
        axs[1].set_ylabel('Degree')
        # axs[0].set_xlabel('Frames')
        axs[0].axes.xaxis.set_visible(False)
        axs[1].set_xlabel('Frames')

        if metric == 'ori':
            axs[0].plot(idx, self.pitch_gt,'b-',label='gt')
            axs[1].plot(idx, self.yaw_gt,'b-',label='gt')

            axs[0].plot(idx, self.pitch_lee_ori,'c-.',label='Lee')
            axs[1].plot(idx, self.yaw_lee_ori,'c-.',label='Lee')


            axs[0].plot(idx, self.pitch_NN_ori,'g--',label='Proposed Method')
            axs[1].plot(idx, self.yaw_NN_ori,'g--',label='Proposed Method')

            axs[0].plot(idx, self.pitch_pau_ori,'r-.',label='Paula')
            axs[1].plot(idx, self.yaw_pau_ori,'r-.',label='Paula')

        if metric == 'vk':
            axs[0].plot(idx, self.pitch_gt,'b-',label='gt')
            axs[1].plot(idx, self.yaw_gt,'b-',label='gt')

            axs[0].plot(idx, self.pitch_lee_vk,'c-.',label='Lee')
            axs[1].plot(idx, self.yaw_lee_vk,'c-.',label='Lee')


            axs[0].plot(idx, self.pitch_NN_vk,'g--',label='Proposed Method')
            axs[1].plot(idx, self.yaw_NN_vk,'g--',label='Proposed Method')

            axs[0].plot(idx, self.pitch_pau_vk,'r-.',label='Paula')
            axs[1].plot(idx, self.yaw_pau_vk,'r-.',label='Paula')

        if metric == 'ok':
            axs[0].plot(idx, self.pitch_gt,'b-',label='gt')
            axs[1].plot(idx, self.yaw_gt,'b-',label='gt')

            axs[0].plot(idx, self.pitch_lee_ok,'c-.',label='Lee')
            axs[1].plot(idx, self.yaw_lee_ok,'c-.',label='Lee')


            axs[0].plot(idx, self.pitch_NN_ok,'g--',label='Proposed Method')
            axs[1].plot(idx, self.yaw_NN_ok,'g--',label='Proposed Method')

            axs[0].plot(idx, self.pitch_pau_ok,'r-.',label='Paula')
            axs[1].plot(idx, self.yaw_pau_ok,'r-.',label='Paula')

        if metric == 'bk':
            axs[0].plot(idx, self.pitch_gt,'b-',label='gt')
            axs[1].plot(idx, self.yaw_gt,'b-',label='gt')

            axs[0].plot(idx, self.pitch_lee_bk,'c-.',label='Lee')
            axs[1].plot(idx, self.yaw_lee_bk,'c-.',label='Lee')


            axs[0].plot(idx, self.pitch_NN_bk,'g--',label='Proposed Method')
            axs[1].plot(idx, self.yaw_NN_bk,'g--',label='Proposed Method')

            axs[0].plot(idx, self.pitch_pau_bk,'r-.',label='Paula')
            axs[1].plot(idx, self.yaw_pau_bk,'r-.',label='Paula')

        # plt.legend()
        axs[0].set_ylim(-11,11)
        axs[1].set_ylim(-11,11)

    def plot_curve_error(self,axs,metric):
        idx = list(range(len(self.pitch_gt)))
        # Set the title for each subplot
        axs[0].set_title('Angle_pitch')
        axs[1].set_title('Angle_yaw')
        axs[0].set_ylabel('Degree')
        axs[1].set_ylabel('Degree')
        # axs[0].set_xlabel('Frames')
        axs[0].axes.xaxis.set_visible(False)
        axs[1].set_xlabel('Frames')

        if metric == 'ori':
            axs[0].plot(idx, self.error_pitch_lee_ori,'c-.',label='Lee')
            axs[1].plot(idx, self.error_yaw_lee_ori,'c-.',label='Lee')


            axs[0].plot(idx, self.error_pitch_NN_ori,'g--',label='Proposed Method')
            axs[1].plot(idx, self.error_yaw_NN_ori,'g--',label='Proposed Method')

            axs[0].plot(idx, self.error_pitch_pau_ori,'r-.',label='Paula')
            axs[1].plot(idx, self.error_yaw_pau_ori,'r-.',label='Paula')

        if metric == 'vk':
            axs[0].plot(idx, self.error_pitch_lee_vk,'c-.',label='Lee')
            axs[1].plot(idx, self.error_yaw_lee_vk,'c-.',label='Lee')


            axs[0].plot(idx, self.error_pitch_NN_vk,'g--',label='Proposed Method')
            axs[1].plot(idx, self.error_yaw_NN_vk,'g--',label='Proposed Method')

            axs[0].plot(idx, self.error_pitch_pau_vk,'r-.',label='Paula')
            axs[1].plot(idx, self.error_yaw_pau_vk,'r-.',label='Paula')

        if metric == 'ok':
            axs[0].plot(idx, self.error_pitch_lee_ok,'c-.',label='Lee')
            axs[1].plot(idx, self.error_yaw_lee_ok,'c-.',label='Lee')


            axs[0].plot(idx, self.error_pitch_NN_ok,'g--',label='Proposed Method')
            axs[1].plot(idx, self.error_yaw_NN_ok,'g--',label='Proposed Method')

            axs[0].plot(idx, self.error_pitch_pau_ok,'r-.',label='Paula')
            axs[1].plot(idx, self.error_yaw_pau_ok,'r-.',label='Paula')

        if metric == 'bk':
            axs[0].plot(idx, self.error_pitch_lee_bk,'c-.',label='Lee')
            axs[1].plot(idx, self.error_yaw_lee_bk,'c-.',label='Lee')


            axs[0].plot(idx, self.error_pitch_NN_bk,'g--',label='Proposed Method')
            axs[1].plot(idx, self.error_yaw_NN_bk,'g--',label='Proposed Method')

            axs[0].plot(idx, self.error_pitch_pau_bk,'r-.',label='Paula')
            axs[1].plot(idx, self.error_yaw_pau_bk,'r-.',label='Paula')

        # plt.legend()
        axs[0].set_ylim(-11,11)
        axs[1].set_ylim(-11,11)
