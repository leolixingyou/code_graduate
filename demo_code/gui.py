import cv2
import copy

from tools import *

class gui:
    def __init__(self):
        self.count = 0
        self.t2 = 0
        self.thresh = 15
        self.stack_pitch = []
        self.stack_yaw = []
        self.flag = False
        self.compelete = False
        self.stop_icon = cv2.resize(cv2.imread('./data/gui/stop.png'),(50,50))
        # self.play_icon = cv2.resize(cv2.imread('./data/gui/play.png'),(50,50))
        self.play_icon = cv2.resize(cv2.imread('./data/gui/yellow_play.png'),(50,50))
        self.pause_icon = cv2.resize(cv2.imread('./data/gui/pause.png'),(50,50))

    def stack_update(self,pitch,yaw):
        self.stack_pitch.append(pitch) 
        self.stack_yaw.append(yaw)
        if len(self.stack_pitch) == self.thresh: 
            self.stack_pitch = self.stack_pitch[1:16]
            self.stack_yaw = self.stack_yaw[1:16]

    def judge(self):
        pitch_max=max(self.stack_pitch)
        pitch_min=min(self.stack_pitch)

        yaw_max=max(self.stack_yaw)
        yaw_min=min(self.stack_yaw)

        if pitch_max-pitch_min < 0.1 and yaw_max-yaw_min<0.1:
            self.flag = False
            self.compelete = True

    def img_pro(self,frame_raw,frame,demo):
        frame_raw = cv2.resize(frame_raw,(1280,720))
        frame = cv2.resize(frame,(1024,512))
        cur_img_raw = copy.copy(frame_raw)
        cur_img = copy.copy(frame)
        cam_height = 2.0  # meter

        field_of_view_deg = 60 # degree
        width, height = frame.shape[1],frame.shape[0]
        intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, width, height)

        pitch_NN_bk, yaw_NN_bk, u_i_k, v_i_k = demo.cal_angle(cur_img,intrinsic_matrix)
        cur_img = self.calibrating(cur_img_raw, pitch_NN_bk, yaw_NN_bk)
        if self.flag:
            self.stack_update(pitch_NN_bk,yaw_NN_bk)
            self.judge()
            cur_img = draw_img(cur_img_raw, pitch_NN_bk, yaw_NN_bk, u_i_k, v_i_k)
            return cur_img, [pitch_NN_bk, yaw_NN_bk]

        else:
            # print('pity')
            return cur_img, [1000,1000]

    def img_pro_demo(self,frame_raw,frame,demo):
        frame_raw = cv2.resize(frame_raw,(1280,720))
        frame = cv2.resize(frame,(1024,512))
        cur_img_raw = copy.copy(frame_raw)
        cur_img = copy.copy(frame)
        cam_height = 2.0  # meter

        field_of_view_deg = 60 # degree
        width, height = frame.shape[1],frame.shape[0]
        intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, width, height)

        pitch_NN_bk, yaw_NN_bk, u_i_k, v_i_k = demo.cal_angle(cur_img,intrinsic_matrix)
        if pitch_NN_bk != None and yaw_NN_bk != None and pitch_NN_bk != [] and yaw_NN_bk != []:
            return cur_img, [pitch_NN_bk, yaw_NN_bk]
        else:
            return cur_img, [1000,1000]

    def residual_error(self,frame_raw,frame,demo):
        frame_raw = cv2.resize(frame_raw,(1280,720))
        frame = cv2.resize(frame,(1024,512))
        cur_img_raw = copy.copy(frame_raw)
        cur_img = copy.copy(frame)
        cam_height = 2.0  # meter

        field_of_view_deg = 60 # degree
        width, height = frame.shape[1],frame.shape[0]
        intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, width, height)

        output_list = demo.error_distribution(cur_img,intrinsic_matrix)
        if output_list != []:
            return cur_img, output_list
        else:
            return cur_img, [1000,1000,1000,1000,[1000,1000],[1000,1000]]

    def calibrating(self,cur_img, pitch_NN_bk, yaw_NN_bk):
        cv2.putText(cur_img,f'FPS: {round(self.t2,2)}',(50,50),0, 1.5, (200,200,0),3,cv2.LINE_AA)
        if pitch_NN_bk != None and yaw_NN_bk != None and not self.flag and not self.compelete:
            cv2.putText(cur_img,f'Avaliable',(50,150),0, 1.5, (200,150,0),3,cv2.LINE_AA)
        elif pitch_NN_bk == None and yaw_NN_bk == None:
            cv2.putText(cur_img,f'Can not Find Lane',(50,150),0, 1.5, (100,100,180),3,cv2.LINE_AA)
        elif self.compelete == True:
            cv2.putText(cur_img,f'Compelet!',(50,150),0, 1.5, (100,180,100),3,cv2.LINE_AA)
        return cur_img

    def create_button(self,frame):
        position = (1180, 50)
        height, width= 50,50

        roi = frame[position[1]:position[1]+height, position[0]:position[0]+width]
        if self.flag == True:
            result = cv2.addWeighted(roi, 1, self.stop_icon, 1, 0)
        if self.flag == False:
            result = cv2.addWeighted(roi, 1, self.play_icon, 1, 0)
        frame[position[1]:position[1]+height, position[0]:position[0]+width] = result

        return frame

