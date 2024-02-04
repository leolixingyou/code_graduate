import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ..tools import *
from .._MLP_models import *

import torch.nn as nn
import torch.optim as optim

NUM_LAYERS = [2,3,5,10,15,20]
FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

def linear_error(gt_p, errors):
    gt_p_array = np.array(gt_p)
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Fit the model to the data
    model.fit(gt_p_array, errors)
    
    # Return the trained model
    return model

def error_cal(gt_a, no_process_list, linear_list, func_list, MLP_list, save_dir):
    no_process_errors = np.mean(np.abs(np.array(no_process_list) - np.array(gt_a)), axis=0)
    print(np.array(func_list).shape, np.array(gt_a).shape)
    linear_errors = np.mean(np.abs(np.squeeze(np.array(linear_list), axis=-1) - np.array(gt_a)), axis=0)
    func_errors = np.mean(np.abs(np.squeeze(np.array(func_list)) - np.array(gt_a)), axis=0)
    MLP_errors = np.mean(np.abs(np.squeeze(np.array(MLP_list)) - np.array(gt_a)), axis=0)
    info_msg = ( f"平均误差: \n \
No Process: Pitch: {no_process_errors[0]:.3f}, Yaw: {MLP_errors[1]:.3f} \n \
Linear: Pitch: {linear_errors[0]:.3f}, Yaw: {linear_errors[1]:.3f}  \n \
Function: Pitch: {func_errors[0]:.3f}, Yaw: {func_errors[1]:.3f}  \n \
MLP: Pitch: {MLP_errors[0]:.3f}, Yaw: {no_process_errors[1]:.3f} \n \
                    ")
    with open (f'{save_dir}error_result.txt','w') as f:
        f.write(info_msg)

def plot_figure(gt_a, no_process_list, linear_list, func_list, MLP_list,save_dir):

    gt_pitch = [item[0] for item in gt_a]
    gt_yaw = [item[1] for item in gt_a]
    sorted_indices = np.argsort(gt_yaw)
    gt_yaw = np.array(gt_yaw)[sorted_indices]
    
    no_process_pitch = [item[0] for item in no_process_list]
    no_process_yaw = np.array([item[1] for item in no_process_list])[sorted_indices]

    linear_pitch = [item[0] for item in linear_list]
    linear_yaw = np.array([item[1] for item in linear_list])[sorted_indices]
    
    func_pitch = [item[0] for item in func_list]
    func_yaw = np.array([item[1] for item in func_list])[sorted_indices]

    MLP_pitch = [item[0] for item in MLP_list]
    MLP_yaw = np.array([item[1] for item in MLP_list])[sorted_indices]

    len_list = int(np.sqrt(len(gt_a)))

    fig, axs = plt.subplots(1, 1, figsize=(12, 9))

    # Plotting pitch values
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), 'b-', label='Ground Truth')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(no_process_pitch,(-1,len_list)),-1), 'c-', marker='v', label='No Process')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(linear_pitch,(-1,len_list)),-1), 'g--', marker='s', label='Linear')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(func_pitch,(-1,len_list)),-1), 'y--', marker='o', label='Function')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(MLP_pitch,(-1,len_list)),-1), 'r-', marker='d', label='MLP')
    axs.set_xlabel('Ground Truth Pitch (degree)')
    axs.set_ylabel('Predicted Pitch (degree)')
    axs.set_title('Comparison of Pitch Values')
    axs.legend()
    axs.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}pitch_comparison.png')  # Save as image file
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=(12, 9))
    # Plotting yaw values
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), 'b-', label='Ground Truth')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(no_process_yaw,(-1,len_list)),-1), 'c-', marker='v', label='No Process')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(linear_yaw,(-1,len_list)),-1), 'g--', marker='s', label='Linear')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(func_yaw,(-1,len_list)),-1), 'y--', marker='o', label='Function')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(MLP_yaw,(-1,len_list)),-1), 'r-', marker='d', label='MLP')
    axs.set_xlabel('Ground Truth Yaw (degree)')
    axs.set_ylabel('Predicted Yaw (degree)')
    axs.set_title('Comparison of Yaw Values')
    axs.legend()
    axs.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}yaw_comparison.png')  # Save as image file
    plt.clf()

    no_process_errors = np.abs(np.array(no_process_list) - np.array(gt_a))
    linear_errors = np.abs(np.squeeze(np.array(linear_list), axis=-1) - np.array(gt_a))
    func_errors = np.abs(np.squeeze(np.array(func_list)) - np.array(gt_a))
    MLP_errors = np.abs(np.squeeze(np.array(MLP_list)) - np.array(gt_a))

    error_lists_pitch = [no_process_errors[:, 0], linear_errors[:, 0], func_errors[:, 0], MLP_errors[:, 0]]
    error_lists_yaw = [no_process_errors[:, 1], linear_errors[:, 1], func_errors[:, 1], MLP_errors[:, 1]]
    
    plot_errors(gt_pitch, error_lists_pitch, "Pitch Errors Comparison", os.path.join(save_dir, 'pitch_errors_comparison.png'),len_list)
    plot_errors(gt_yaw, error_lists_yaw, "Yaw Errors Comparison", os.path.join(save_dir, 'yaw_errors_comparison.png'),len_list)

def plot_errors(gt_a, errors, title, save_path, len_list):
    fig, ax = plt.subplots(figsize=(12, 9))
    if 'Pitch ' in title:
        ax.plot(gt_a, [0]*len(gt_a), 'b-', label='Ground Truth', alpha=0.8)
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[0],(-1,len_list)),-1), 'c-', marker='v', label='No Process')
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[1],(-1,len_list)),-1), 'g--', marker='s', label='Linear')
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[2],(-1,len_list)),-1), 'y--', marker='o', label='Function')
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[3],(-1,len_list)),-1), 'r-', marker='d', label='MLP')
    else:
        ax.plot(gt_a, [0]*len(gt_a), 'b-', label='Ground Truth', alpha=0.8)
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[3],(-1,len_list)),-1), 'c-', marker='v', label='No Process')
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[1],(-1,len_list)),-1), 'g--', marker='s', label='Linear')
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[2],(-1,len_list)),-1), 'y--', marker='o', label='Function')
        ax.plot(np.mean(np.reshape(gt_a,(-1,len_list)),-1), np.mean(np.reshape(errors[0],(-1,len_list)),-1), 'r-', marker='d', label='MLP')
    ax.set_xlabel('Ground Truth Value (degree)')
    ax.set_ylabel('Error (degree)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

def plane_f(gt_a,num,gt_p,error_list):
    p1,p2,p3,p4 = 0,0,0,0
    for id, con in enumerate(gt_a):
        if con == [num,num]:
            p1 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
        if con == [num,-num]:
            p2 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
        if con == [-num,-num]:
            p3 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
        if con == [-num,num]:
            p4 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
    v1 = p3 - p1
    v2 = p2 - p1
    norm_p = np.cross(v1, v2)
    d = np.dot(norm_p, p1)
    if norm_p[2] != 0:
        plane_func = lambda x, y: (-norm_p[0]*x - norm_p[1]*y - d) / norm_p[2]
    else:
        v1 = p3 - p4
        v2 = p2 - p4
        norm_p = np.cross(v1, v2)
        d = np.dot(norm_p, p1)
        plane_func = lambda x, y: (-norm_p[0]*x - norm_p[1]*y - d) / norm_p[2]
    return plane_func


class error_conpensation:
    def __init__(self,field_of_view_deg,save_dir):
        cam_height = 3.0  # meter
        width,height = 1024,512

        self.save_dir = save_dir
        make_dir(save_dir)

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

    def train(self, gt_a,gt_p):
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
        num_layer = NUM_LAYERS[2]
        net = judge_layers(num_layer)

        # 定义损失函数和优化器
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # 训练神经网络
        num_epochs = 50
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
        # torch.save(net.state_dict(), f"./weights/layers_{num_layer}_pitch_yaw_model_231009.pth")###### EDIT HERE FOV60
        torch.save(net.state_dict(), f"./weights/layers_{num_layer}_pitch_yaw_model_fov30.pth")###### EDIT HERE
        print(f"best Loss is {min(loss_list)}")
        print(f'Average time is {(sum(time_list))/len(time_list)}')

    def validation_cal(self, gt_a,gt_p,pitch_f, yaw_f, pitch_linear, yaw_linear,save_dir):
        ### validattion
        model = ErrorPredictionNN_5()
        model.load_state_dict(torch.load("./weights/layers_5_pitch_yaw_model_fov30.pth"))###### EDIT HERE
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

            pitch_i, yaw_i = [np.rad2deg(x) for x in get_py_from_vp(u_gt, v_gt, self.intrinsic_matricx)]
            
            new_vp2cp = vector_from_vp_to_cp(vp,cp)
            new_vp2cp_tensor = torch.tensor(new_vp2cp, dtype=torch.float32)
            
            ### MLP method
            with torch.no_grad():
                predicted_output = model(new_vp2cp_tensor)
                predicted_pitch_errors, predicted_yaw_errors = predicted_output[0].numpy(), predicted_output[1].numpy()
            pitch_NN_ori, yaw_NN_ori = pitch_i + predicted_pitch_errors, yaw_i + predicted_yaw_errors

            ### linear method
            pitch_linear_error = pitch_linear.predict([new_vp2cp])
            yaw_linear_error = yaw_linear.predict([new_vp2cp])
            pitch_l, yaw_l = pitch_i - pitch_linear_error, yaw_i - yaw_linear_error

            ### plane function method
            error_pitch = pitch_f(u_gt,v_gt)
            error_yaw = yaw_f(u_gt,v_gt)
            pitch_c, yaw_c = pitch_i + error_pitch, yaw_i + error_yaw

            no_process_list.append([pitch_i, yaw_i])
            linear_list.append([pitch_l, yaw_l])
            func_list.append([pitch_c, yaw_c])
            MLP_list.append([pitch_NN_ori, yaw_NN_ori])
            # print(gt_a[id_v][0],gt_a[id_v][1])
            # print(pitch_i, yaw_i)
            # print(pitch_l, yaw_l)
            # print(pitch_c, yaw_c)
            # print(pitch_NN_ori, yaw_NN_ori)

        return [gt_a, no_process_list, linear_list, func_list, MLP_list,save_dir]

    def make_data(self, num):
        gt_a,gt_p = self.make_gt(num)
        pitch_f, yaw_f, pitch_linaer, yaw_linaer = self.calcu_error(gt_a,gt_p,num)
        return gt_a, gt_p, pitch_f, yaw_f, pitch_linaer, yaw_linaer

    def main(self, mode, angle_num):
        save_dir = self.save_dir
        gt_a, gt_p, pitch_f, yaw_f, pitch_linaer, yaw_linaer = self.make_data(angle_num)

        ######## MLP
        if mode == 'train':
            self.train(gt_a,gt_p)
        
        ######## Validation
        if mode == 'validation':
            return self.validation_cal(gt_a,gt_p,pitch_f, yaw_f, pitch_linaer, yaw_linaer,save_dir)

def plot_figure_original(original_list, save_dir):

    colors = ['c', 'g', 'm', 'y', 'k']
    markers = ['v', 'o', 's', 'p', 'D']
    save_dir = save_dir + '/../'

    
    ### set gt_pitch & gt_yaw
    for idx, (gt_a, result, flag) in enumerate(original_list):
        gt_pitch = [item[0] for item in gt_a]
        len_list = int(np.sqrt(len(gt_a)))
        gt_yaw = [item[1] for item in gt_a]
        sorted_indices = np.argsort(gt_yaw)
        gt_yaw = np.array(gt_yaw)[sorted_indices]
        
        process_errors = np.abs(np.array(result) - np.array(gt_a))

    # Pitch plot
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 9))
    axs1[0].plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), 'b-', label='Ground Truth')
    axs1[1].plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), [0]*len(np.reshape(gt_pitch,(-1,len_list))), 'b-', label='Ground Truth')
    for idx, (gt_a, result, flag) in enumerate(original_list):
        processed_pitch = [item[0] for item in result]

        axs1[0].plot(np.mean(np.reshape(gt_pitch, (-1, len_list)), -1), np.mean(np.reshape(processed_pitch, (-1, len_list)), -1), color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)
        axs1[1].plot(np.mean(np.reshape(gt_pitch, (-1, len_list)), -1), np.mean(np.reshape(process_errors[:, 0], (-1, len_list)), -1), color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)

    axs1[0].set_xlabel('Ground Truth Pitch (degree)')
    axs1[0].set_ylabel('Predicted Pitch (degree)')
    axs1[0].set_title('Comparison of Pitch Values with FOV')
    axs1[0].legend()
    axs1[0].grid(True)
    plt.tight_layout()

    axs1[1].set_xlabel('Ground Truth Pitch (degree)')
    axs1[1].set_ylabel('Predicted Pitch error (degree)')
    axs1[1].set_title('Comparison of Pitch error with FOV')
    axs1[1].legend()
    axs1[1].grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}Pitch Performance with FOV.png')
    plt.clf()

    # Yaw plot
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 9))
    axs2[0].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), 'b-', label='Ground Truth')
    axs2[1].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), [0]*len(np.reshape(gt_yaw,(-1,len_list))), 'b-', label='Ground Truth')
    for idx, (gt_a, result, flag) in enumerate(original_list):
        processed_yaw = np.array([item[1] for item in result])[sorted_indices]
        processed_yaw_avg = np.mean(np.reshape(processed_yaw, (-1, len_list)), -1)
        axs2[0].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), processed_yaw_avg, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)
        axs2[1].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), np.mean(np.reshape(process_errors[:, 1], (-1, len_list)), -1), color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)

    axs2[0].set_xlabel('Ground Truth Yaw (degree)')
    axs2[0].set_ylabel('Predicted Yaw (degree)')
    axs2[0].set_title('Comparison of Yaw Values with FOV')
    axs2[0].legend()
    axs2[0].grid(True)
    plt.tight_layout()

    axs2[1].set_xlabel('Ground Truth Yaw (degree)')
    axs2[1].set_ylabel('Predicted Yaw error (degree)')
    axs2[1].set_title('Comparison of Yaw error with FOV')
    axs2[1].legend()
    axs2[1].grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}Yaw Performance with FOV.png')
    plt.clf()

def postprocess(result_list):
    original_list = []
    save_dir_original = None
    # ### regular results with compensation part
    for results,flag in result_list:
        print(len(results), flag)
        gt_a, no_process_list, linear_list, func_list, MLP_list,save_dir = results
        error_cal(gt_a, no_process_list, linear_list, func_list, MLP_list, save_dir)
        plot_figure(gt_a, no_process_list, linear_list, func_list, MLP_list,save_dir)
        save_dir_original = save_dir
        original_list.append([gt_a, no_process_list,flag])
    ### original results with different FOV
    plot_figure_original(original_list, save_dir_original)

if __name__ == "__main__":
    field_of_view_deg_list = [30,60,90,120,150,180]
    field_of_view_deg_list = [60,90,120,150]### usable range
    # field_of_view_deg_list = [30]
    mode_list = ['train','validation']
    mode = mode_list[1]
    angle_num= 30

    result_list = []
    for field_of_view_deg in field_of_view_deg_list:
        save_dir = f'../results/residual_error/IMG_{FOLDER_NAME}/FOV_{field_of_view_deg}/'
        ErrorCon = error_conpensation(field_of_view_deg, save_dir)
        # plt.show()
        result_list.append([ErrorCon.main(mode, angle_num),field_of_view_deg])
    
    postprocess(result_list)
