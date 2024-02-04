import json
import time
import numpy as np

from plots_img import *
from error_models import *

FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

def error_cal(gt_a, no_process_list, linear_list, func_list, MLP_list, save_dir):
    no_process_list_new = np.array(no_process_list)[:,:2]
    linear_list_new = np.array(linear_list)[:,:2]
    func_list_new = np.array(func_list)[:,:2]
    MLP_list_new = np.array(MLP_list)[:,:2]

    no_process_errors = np.mean(np.abs(no_process_list_new - np.array(gt_a)), axis=0)
    linear_errors = np.mean(np.abs(np.squeeze(np.array(linear_list_new)[:,:2]) - np.array(gt_a)), axis=0)
    func_errors = np.mean(np.abs(np.squeeze(np.array(func_list_new)[:,:2]) - np.array(gt_a)), axis=0)
    MLP_errors = np.mean(np.abs(np.squeeze(np.array(MLP_list_new)[:,:2]) - np.array(gt_a)), axis=0)

    no_process_fps = 1000 * float(np.mean(np.abs(np.array(no_process_list)[:,2]), axis=0))
    linear_fps = 1000 * float(np.mean(np.abs(np.array(linear_list)[:,2]), axis=0))
    func_fps = 1000 * float(np.mean(np.abs(np.array(func_list)[:,2]), axis=0))
    MLP_fps = 1000 * float(np.mean(np.abs(np.array(MLP_list)[:,2]), axis=0))

    info_msg = ( f"Map: \n \
No Process: Pitch: {no_process_errors[0]:.3f}, Yaw: {MLP_errors[1]:.3f}, FPS: {no_process_fps:.3f} \n \
Linear: Pitch: {linear_errors[0][0]:.3f}, Yaw: {linear_errors[1][0]:.3f}, FPS: {linear_fps:.3f}  \n \
Function: Pitch: {func_errors[0]:.3f}, Yaw: {func_errors[1]:.3f}, FPS: {func_fps:.3f}  \n \
MLP: Pitch: {MLP_errors[0]:.3f}, Yaw: {no_process_errors[1]:.3f}, FPS: {MLP_fps:.3f} \n ")
    with open (f'{save_dir}error_result.txt','w') as f:
        f.write(info_msg)
    return no_process_list_new, linear_list_new, func_list_new, MLP_list_new

def reset_list(src_list, tar_list):
    src_list = list(src_list)
    tar_list = list(tar_list)
    src_list_sorted = sorted(src_list)
    tar_list_sorted = [tar_list[src_list.index(x)]  for x in src_list_sorted]
    return tar_list_sorted

def postprocess(result_list, figure_size, font_size, field_of_view_deg_list, NUM_LAYERS, mode):
    if mode == 'train':
        result_list = np.array(result_list)
        fov_list = result_list[:,0]
        num_layer = result_list[:,1]
        loss_list = result_list[:,2]
        avg_list = result_list[:,3]
        save_dir = result_list[:,4][0]

        
        plot_train_figure(fov_list, num_layer, loss_list, avg_list, save_dir, figure_size, font_size)
        with open(f'{save_dir}avg_time.txt', 'w')as f :
            f.write(str(np.stack((fov_list,num_layer,avg_list),axis=1)))

    if mode == 'validation':
        original_list = []
        MLP_errors_list = []
        save_dir_original = None
        # ### regular results with compensation part
        for results,flag in result_list:
            print(len(results), flag)
            gt_a, no_process_list, linear_list, func_list, MLP_list,save_dir = results
            no_process_list, linear_list, func_list, MLP_list = error_cal(gt_a, no_process_list, linear_list, func_list, MLP_list, save_dir)
            plot_figure(gt_a, no_process_list, linear_list, func_list, MLP_list, save_dir, figure_size, font_size)
            save_dir_original = save_dir
            original_list.append([gt_a, MLP_list,flag])
        ### original results with different FOV
        plot_figure_original(original_list, save_dir_original, figure_size, font_size)

if __name__ == "__main__":
    field_of_view_deg_list = [30,60,90,120,150,180]
    field_of_view_deg_list = [60,90,120,150]### usable range
    NUM_LAYERS = [2,3,5,10,15,20]
    # field_of_view_deg_list = [30]
    mode_list = ['train','validation']
    mode = mode_list[0]
    angle_num= 30
    figure_size = (18,9)
    font_size = 25
    num_epochs = 50
    # num_layer = NUM_LAYERS[2]
    ## save check points
    save_che = False

    using_mlp_path = "../weights/layers_5_pitch_yaw_model_pr.pth"
    save_mlp_path = f"../weights/{time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))}/"
    make_dir(save_mlp_path)

    result_json = './training_result.json'
    flag = os.path.exists(result_json)
    result_list = []
    if not flag:
        count_layer = 0
        count_fov = 0
        save_dir = f'../results/residual_error/IMG_{FOLDER_NAME}/'
        for field_of_view_deg in field_of_view_deg_list:
            ErrorCon = error_conpensation(field_of_view_deg, save_dir)
            if mode == 'train':
                for num_layer in NUM_LAYERS:
                    print(f'field_of_view_deg is {field_of_view_deg}')
                    print(f'layer is {num_layer}')
                    mlp_path = save_mlp_path
                    result_list.append(ErrorCon.main(mode, angle_num, mlp_path, num_layer, num_epochs, save_che))
                #     count_layer += 1
                #     if count_layer == 3:
                #         count_layer =0
                #         break
                # count_fov += 1
                # if count_fov == 3:
                #     count_fov =0
                #     break
            if mode == 'validation':
                save_dir = f'{save_dir}FOV_{field_of_view_deg}/'
                mlp_path = using_mlp_path
                result_list.append([ErrorCon.main(mode, angle_num, mlp_path,[], [], [],[]),field_of_view_deg])
        save_result = {'haha': result_list}
        with open(result_json, 'w') as f:
            json.dump(save_result,f)
    else:
        print(1)
        with open(result_json, 'r') as f:
            result_list = json.load(f)['haha']
    postprocess(result_list, figure_size, font_size, field_of_view_deg_list, NUM_LAYERS, mode)
