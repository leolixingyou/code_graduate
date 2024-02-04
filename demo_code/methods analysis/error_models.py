import time
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim

from methods_set import *
from tools import *
from _MLP_models import *

NUM_LAYERS = [2,3,5,10,15,20]

class error_conpensation:
    def __init__(self,field_of_view_deg,save_dir):
        cam_height = 3.0  # meter
        width,height = 1024,512

        self.save_dir = save_dir
        make_dir(save_dir)
        self.fov = field_of_view_deg
        self.intrinsic_matricx = get_intrinsic_matrix(field_of_view_deg, 1024, 512)
        
    def make_gt(self, num):
        manx = num+1
        minx = -num
        py_gt = np.arange(minx,manx,1)

        gt_a = []
        gt_p = []
        for pitch in py_gt:
            for yaw in py_gt:  
                u_gt, v_gt = get_vp_from_py(pitch, yaw, self.intrinsic_matricx)
                gt_p.append([u_gt,v_gt])
                gt_a.append([pitch,yaw])
        return gt_a,gt_p
    
    def calcu_error(self, gt_a,gt_p,num):
        ### calculation
        pitch_error_list = []
        yaw_error_list = []
        for id,vp in enumerate(gt_p):
            u_gt,v_gt = vp
            pitch_i, yaw_i = [np.rad2deg(x) for x in get_py_from_vp(u_gt, v_gt, self.intrinsic_matricx)]
            pitch_error, yaw_error = gt_a[id][0] - pitch_i, gt_a[id][1] - yaw_i
            pitch_error_list.append(pitch_error)
            yaw_error_list.append(yaw_error)
        
        pitch_linear = linear_error(gt_p, pitch_error_list)
        yaw_linear = linear_error(gt_p, yaw_error_list)

        pitch_f = plane_f(gt_a, num,gt_p, pitch_error_list)
        yaw_f = plane_f(gt_a, num,gt_p, yaw_error_list)

        return pitch_f, yaw_f, pitch_linear, yaw_linear

    def train(self, gt_a,gt_p, mlp_path, num_layer, num_epochs, save_dir,save_che):
        ### calculation
        pitch_error_list = []
        yaw_error_list = []
        for id,vp in enumerate(gt_p):
            u_gt,v_gt = vp
            pitch_i, yaw_i = [np.rad2deg(x) for x in get_py_from_vp(u_gt, v_gt, self.intrinsic_matricx)]
            pitch_error, yaw_error = gt_a[id][0] - pitch_i, gt_a[id][1] - yaw_i
            pitch_error_list.append(pitch_error)
            yaw_error_list.append(yaw_error)
        
        gt_cp = [[1024/2,512/2]]*len(gt_a)
        vp = np.array(gt_p)
        cp = np.array(gt_cp)
        vp2cp = vector_from_vp_to_cp(vp,cp)
        new_pitch_error = np.array(pitch_error_list)
        new_yaw_error = np.array(yaw_error_list)

        vp2cp_tensor = torch.tensor(vp2cp, dtype=torch.float32)
        pitch_error_tensor = torch.tensor(new_pitch_error, dtype=torch.float32)
        yaw_error_tensor = torch.tensor(new_yaw_error, dtype=torch.float32)

        print(vp2cp_tensor.shape)
        # 将数据拼接为一个张量
        output_data = torch.stack((pitch_error_tensor, yaw_error_tensor), dim=1)

        # 创建神经网络实例
        net = judge_layers(num_layer)

        # 定义损失函数和优化器
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # 训练神经网络
        loss_list = []
        time_list = []
        for epoch in range(num_epochs):
            # 前向传播
            t1 = time.time()
            predicted_output = net(vp2cp_tensor)

            # 计算损失
            loss = loss_function(predicted_output, output_data)
            print(f'Epoch_{epoch}: {loss}')
            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新权重
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                loss_list.append(loss.item())
            time_list.append(time.time()-t1)
        # 保存训练好的模型
        train_check_point = f"{mlp_path}layers_{num_layer}_pitch_yaw_model_fov{self.fov}.pth"  
        # torch.save(net.state_dict(), f"./weights/layers_{num_layer}_pitch_yaw_model_231009.pth")###### EDIT HERE FOV60
        if save_che:
            torch.save(net.state_dict(), train_check_point)###### EDIT HERE mlp_path
        print(f"best Loss is {min(loss_list)}")
        print(f'Average time is {(sum(time_list))/len(time_list)}')
        return [self.fov, num_layer, loss_list, (sum(time_list))/len(time_list),save_dir]

    def validation_cal(self, gt_a,gt_p,pitch_f, yaw_f, pitch_linear, yaw_linear, save_dir, mlp_path):
        ### validattion
        model = ErrorPredictionNN_5()
        model.load_state_dict(torch.load(mlp_path))###### EDIT HERE
        # model.load_state_dict(torch.load("./weights/layers_5_pitch_yaw_model_231009.pth"))###### EDIT HERE FOV60
        model.eval()

        no_process_list = []
        linear_list = []
        func_list = []
        MLP_list = []
        
        for id_v,vp_v in enumerate(gt_p):
            u_gt, v_gt = vp_v
            vp = np.array([u_gt,v_gt])
            cp = np.array([1024/2,512/2])
            
            lee_start_time = time.time()
            pitch_i, yaw_i = [np.rad2deg(x) for x in get_py_from_vp(u_gt, v_gt, self.intrinsic_matricx)]
            lee_end_time = lee_start_time - time.time() 
            
            compensate_start_time = time.time()
            new_vp2cp = vector_from_vp_to_cp(vp,cp)
            new_vp2cp_tensor = torch.tensor(new_vp2cp, dtype=torch.float32)
            compensate_end_time = compensate_start_time - time.time()
            
            mlp_start_time = time.time()
            ### MLP method
            with torch.no_grad():
                predicted_output = model(new_vp2cp_tensor)
                predicted_pitch_errors, predicted_yaw_errors = predicted_output[0].numpy(), predicted_output[1].numpy()
            pitch_NN_ori, yaw_NN_ori = pitch_i + predicted_pitch_errors, yaw_i + predicted_yaw_errors
            mlp_end_time = mlp_start_time - time.time() 

            linear_start_time = time.time()
            ### linear method
            pitch_linear_error = pitch_linear.predict([new_vp2cp])
            yaw_linear_error = yaw_linear.predict([new_vp2cp])
            pitch_l, yaw_l = pitch_i - pitch_linear_error, yaw_i - yaw_linear_error
            linear_end_time = linear_start_time - time.time() 

            plane_start_time = time.time()
            ### plane function method
            error_pitch = pitch_f(u_gt,v_gt)
            error_yaw = yaw_f(u_gt,v_gt)
            pitch_c, yaw_c = pitch_i + error_pitch, yaw_i + error_yaw
            plane_end_time = plane_start_time - time.time() 

            no_process_list.append([pitch_i, yaw_i, lee_end_time])
            linear_list.append([pitch_l, yaw_l, linear_end_time])
            func_list.append([pitch_c, yaw_c, plane_end_time])
            MLP_list.append([pitch_NN_ori, yaw_NN_ori, mlp_end_time])

        return [gt_a, no_process_list, linear_list, func_list, MLP_list,save_dir]

    def make_data(self, num):
        gt_a,gt_p = self.make_gt(num)
        pitch_f, yaw_f, pitch_linaer, yaw_linaer = self.calcu_error(gt_a,gt_p,num)
        return gt_a, gt_p, pitch_f, yaw_f, pitch_linaer, yaw_linaer

    def main(self, mode, angle_num, mlp_path, num_layer, num_epochs, save_che):
        save_dir = self.save_dir
        gt_a, gt_p, pitch_f, yaw_f, pitch_linaer, yaw_linaer = self.make_data(angle_num)

        ######## Train MLP
        if mode == 'train':
            return self.train(gt_a,gt_p, mlp_path, num_layer, num_epochs, save_dir,save_che)

        ######## Validation
        if mode == 'validation':
            return self.validation_cal(gt_a,gt_p,pitch_f, yaw_f, pitch_linaer, yaw_linaer,save_dir, mlp_path)
