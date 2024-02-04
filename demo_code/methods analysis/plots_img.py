import os
import numpy as np
import matplotlib.pyplot as plt

def plot_train_figure(fov_list, num_layer, loss_list, avg_list, save_dir, figure_size, font_size):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['v', 'o', 's', 'p', 'D']

    layer_indx  = [np.where(num_layer==x) for x in set(num_layer)]
    fov_indx = [np.where(fov_list==x) for x in set(fov_list)]

    color_list = []
    marker_list = []
    for x in range(len(loss_list)):
        layer_indx_list = [list(y[0]) for y in layer_indx]
        fov_indx_list = [list(y[0]) for y in fov_indx]
        color_list.append(colors[next((i for i, sublist in enumerate(layer_indx_list) if x in sublist), None)])
        marker_list.append(markers[next((i for i, sublist in enumerate(fov_indx_list) if x in sublist), None)])

    ### set gt_pitch & gt_yaw
    fig, axs = plt.subplots(1, 1, figsize=figure_size)
    for i, loss in enumerate(loss_list):
        # Plotting pitch values
        axs.plot([x *10 for x in range(1,len(loss)+1)], loss, f'{color_list[i]}{marker_list[i]}', label=f'Layer {num_layer[i]} FOV {fov_list[i]}')

    axs.set_xlabel('Epoch', fontsize = font_size)
    axs.set_ylabel('Loss', fontsize = font_size)
    axs.set_ylim(0, 1)
    axs.set_title('MLP Loss Curve with FOV and Layer Number ', fontsize = font_size)
    axs.tick_params(labelsize = font_size)
    axs.grid(True)
    # axs.legend(fontsize = 15,ncol=2)
    plt.tight_layout()
    plt.savefig(f'{save_dir}Loss_curve_no_legend.svg',dpi=300)  # Save as image file
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=figure_size)
    for i, loss in enumerate(loss_list):
        # Plotting pitch values
        axs.plot([x *10 for x in range(1,len(loss)+1)], loss, f'{color_list[i]}{marker_list[i]}', label=f'Layer {num_layer[i]} FOV {fov_list[i]}')

    axs.set_xlabel('Epoch', fontsize = font_size)
    axs.set_ylabel('Loss', fontsize = font_size)
    axs.set_title('MLP Loss Curve with FOV and Layer Number ', fontsize = font_size)
    axs.tick_params(labelsize = font_size)
    axs.grid(True)
    # axs.legend(fontsize = 15,ncol=2)
    plt.tight_layout()
    plt.savefig(f'{save_dir}Loss_curve_no_limit_no_legend.svg',dpi=300)  # Save as image file
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=figure_size)
    avg_avg_list = [round(x,4) for x in np.mean(np.reshape(avg_list,(4,-1)),axis=0)]
    num_num_layer = list(set(num_layer))

    for  i in range(len(avg_avg_list)-2):
        pi = axs.bar(num_num_layer[i], avg_avg_list[i], fc=colors[i],label=f'{avg_avg_list[i]}')
        axs.bar_label(pi)

    axs.set_xticks(np.arange(1, max(num_num_layer)+1, step=1))
    axs.set_xlabel('Layer', fontsize = font_size)
    axs.set_ylabel('Time (s)', fontsize = font_size)
    axs.set_title('Inferencing Time of Different Depth Networks', fontsize = font_size)
    axs.tick_params(labelsize = font_size)
    axs.grid(True)
    axs.legend(fontsize = font_size,ncol=2)
    plt.tight_layout()
    plt.savefig(f'{save_dir}time_avg.svg',dpi=300)  # Save as image file
    plt.clf()



def plot_figure(gt_a, no_process_list, linear_list, func_list, MLP_list, save_dir, figure_size, font_size):

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

    fig, axs = plt.subplots(1, 1, figsize=figure_size)

    # Plotting pitch values
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), 'b-', label='Ground Truth')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(no_process_pitch,(-1,len_list)),-1), 'c-', marker='v', label='No Process')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(linear_pitch,(-1,len_list)),-1), 'g--', marker='s', label='Linear')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(func_pitch,(-1,len_list)),-1), 'y--', marker='o', label='Function')
    axs.plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(MLP_pitch,(-1,len_list)),-1), 'r-', marker='d', label='MLP')
    axs.set_xlabel('Ground Truth Pitch (°)', fontsize = font_size)
    axs.set_ylabel('Predicted Pitch (°)', fontsize = font_size)
    axs.set_title('Comparison of Pitch Values', fontsize = font_size)
    # axs.legend(fontsize = font_size)
    axs.tick_params(labelsize = font_size)
    axs.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}pitch_comparison.svg',dpi=300)  # Save as image file
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=figure_size)
    # Plotting yaw values
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), 'b-', label='Ground Truth')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(no_process_yaw,(-1,len_list)),-1), 'c-', marker='v', label='No Process')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(linear_yaw,(-1,len_list)),-1), 'g--', marker='s', label='Linear')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(func_yaw,(-1,len_list)),-1), 'y--', marker='o', label='Function')
    axs.plot(np.mean(np.reshape(gt_yaw,(-1,len_list)),-1), np.mean(np.reshape(MLP_yaw,(-1,len_list)),-1), 'r-', marker='d', label='MLP')
    axs.set_xlabel('Ground Truth Yaw (°)', fontsize = font_size)
    axs.set_ylabel('Predicted Yaw (°)', fontsize = font_size)
    axs.set_title('Comparison of Yaw Values', fontsize = font_size)
    # axs.legend(fontsize = font_size)
    axs.tick_params(labelsize = font_size)
    axs.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}yaw_comparison.svg',dpi=300)  # Save as image file
    plt.clf()

    no_process_errors = np.abs(np.array(no_process_list)[:,:2] - np.array(gt_a))
    linear_errors = np.abs(np.squeeze(np.array(linear_list)[:,:2]) - np.array(gt_a))
    func_errors = np.abs(np.squeeze(np.array(func_list)[:,:2]) - np.array(gt_a))
    MLP_errors = np.abs(np.squeeze(np.array(MLP_list)[:,:2]) - np.array(gt_a))

    error_lists_pitch = [no_process_errors[:, 0], linear_errors[:, 0], func_errors[:, 0], MLP_errors[:, 0]]
    error_lists_yaw = [no_process_errors[:, 1], linear_errors[:, 1], func_errors[:, 1], MLP_errors[:, 1]]
    
    plot_errors(gt_pitch, error_lists_pitch, "Pitch Errors Comparison", os.path.join(save_dir, 'pitch_errors_comparison.svg'),len_list,figure_size,font_size)
    plot_errors(gt_yaw, error_lists_yaw, "Yaw Errors Comparison", os.path.join(save_dir, 'yaw_errors_comparison.svg'),len_list,figure_size,font_size)
    
    return MLP_errors 

def plot_errors(gt_a, errors, title, save_path, len_list, figure_size, font_size):
    fig, ax = plt.subplots(figsize=figure_size)
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
    ax.set_xlabel('Ground Truth Value (°)', fontsize = font_size)
    ax.set_ylabel('Error (°)', fontsize = font_size)
    ax.tick_params(labelsize = font_size)
    ax.set_title(title)
    # ax.legend(fontsize = font_size)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)
    plt.clf()

def plot_figure_original(original_list, save_dir, figure_size, font_size):

    colors = ['c', 'g', 'm', 'y', 'k']
    markers = ['v', 'o', 's', 'p', 'D']
    save_dir = save_dir + '/../'
    process_errors_list = []
    ### set gt_pitch & gt_yaw
    for idx, (gt_a, result, flag) in enumerate(original_list):
        gt_pitch = [item[0] for item in gt_a]
        len_list = int(np.sqrt(len(gt_a)))
        gt_yaw = [item[1] for item in gt_a]
        sorted_indices = np.argsort(gt_yaw)
        gt_yaw = np.array(gt_yaw)[sorted_indices]
        process_errors = np.abs(np.squeeze(np.array(result)[:,:2]) - np.array(gt_a))
        process_errors_list.append(process_errors)
        # process_errors = np.abs(np.squeeze(np.array(result)[:,:2]) - np.array(gt_a))

    # Pitch plot
    fig1, axs1 = plt.subplots(2, 1, figsize=figure_size)
    # axs1[0].plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), 'b-', label='Ground Truth')
    # axs1[1].plot(np.mean(np.reshape(gt_pitch,(-1,len_list)),-1), [0]*len(np.reshape(gt_pitch,(-1,len_list))), 'b-', label='Ground Truth')
    for idx, (gt_a, result, flag) in enumerate(original_list):
        processed_pitch = [item[0] for item in result]

        axs1[0].plot(np.mean(np.reshape(gt_pitch, (-1, len_list)), -1), np.mean(np.reshape(processed_pitch, (-1, len_list)), -1), color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)
        axs1[1].plot(np.mean(np.reshape(gt_pitch, (-1, len_list)), -1), np.mean(np.reshape(process_errors_list[idx][:, 0], (-1, len_list)), -1), color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)

    axs1[0].set_xlabel('Ground Truth Pitch (°)', fontsize = font_size)
    axs1[0].set_ylabel('Predicted Pitch (°)', fontsize = font_size)
    axs1[0].set_title('Comparison of Pitch Values with FOV')
    # axs1[0].legend(fontsize = font_size)
    axs1[0].tick_params(labelsize = font_size)
    axs1[0].grid(True)
    plt.tight_layout()

    axs1[1].set_xlabel('Ground Truth Pitch (°)', fontsize = font_size)
    axs1[1].set_ylabel('Predicted Pitch error (°)', fontsize = font_size)
    axs1[1].set_title('Comparison of Pitch error with FOV')
    # axs1[1].legend(fontsize = font_size)
    axs1[1].tick_params(labelsize = font_size)
    axs1[1].grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}Pitch Performance FOV.svg',dpi=300)
    plt.clf()

    # Yaw plot
    fig2, axs2 = plt.subplots(2, 1, figsize=figure_size)
    # axs2[0].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), 'b-', label='Ground Truth')
    # axs2[1].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), [0]*len(np.reshape(gt_yaw,(-1,len_list))), 'b-', label='Ground Truth')
    for idx, (gt_a, result, flag) in enumerate(original_list):
        processed_yaw = np.array([item[1] for item in result])[sorted_indices]
        processed_yaw_avg = np.mean(np.reshape(processed_yaw, (-1, len_list)), -1)
        axs2[0].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), processed_yaw_avg, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)
        axs2[1].plot(np.mean(np.reshape(gt_yaw, (-1, len_list)), -1), np.mean(np.reshape(process_errors_list[idx][:, 1], (-1, len_list)), -1), color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=f'FOV_{flag}', alpha = 0.5)

    axs2[0].set_xlabel('Ground Truth Yaw (°)', fontsize = font_size)
    axs2[0].set_ylabel('Predicted Yaw (°)', fontsize = font_size)
    axs2[0].set_title('Comparison of Yaw Values with FOV')
    # axs2[0].legend(fontsize = font_size)
    axs2[0].tick_params(labelsize = font_size)
    axs2[0].grid(True)
    plt.tight_layout()

    axs2[1].set_xlabel('Ground Truth Yaw (°)', fontsize = font_size)
    axs2[1].set_ylabel('Predicted Yaw error (°)', fontsize = font_size)
    axs2[1].set_title('Comparison of Yaw error with FOV')
    # axs2[1].legend(fontsize = font_size)
    axs2[1].tick_params(labelsize = font_size)
    axs2[1].grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}Yaw Performance with FOV.svg',dpi=300)
    plt.clf()