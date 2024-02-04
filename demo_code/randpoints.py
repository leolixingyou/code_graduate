import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

def plot_rand(flag,bias):
    # 区间数
    num_intervals = 10

    # 区间长度
    interval_length = 5

    # 生成数据
    gt_values = []
    predicted_values = []

    for i in range(1, num_intervals + 1):
        # 区间起点和终点
        interval_start = i * interval_length
        interval_end = (i + 1) * interval_length

        # 生成当前区间内的随机点
        # num_points = np.random.randint(18, 20)  # 每个区间生成5到10个随机点
        if num_intervals > 5:
            num_points = 5  # 每个区间生成5到10个随机点
        else:
            num_points = 20  # 每个区间生成5到10个随机点

        for _ in range(num_points):
            # 生成满足条件的随机点
            x = np.random.uniform(interval_start, interval_end)
            a_range = (0.01 + 0.02*bias) + (0.02 * (i-1))  # a的范围
            if flag== 'look_down':
                a = np.random.uniform(0,a_range)
                y = x + (x * a)
            else:
                a = np.random.uniform(-a_range, 0)
                y = x + (x * a)
            with open(f'{save_dir}{flag}.csv', "a") as f1:
                csv_writer = csv.writer(f1)
                csv_writer.writerow([
                    x, y
                                ])
            gt_values.append(x)
            predicted_values.append(y)
    return gt_values, predicted_values

def calculate_linear_least_squares(gt_values, predicted_values):
    # Convert the lists of values into numpy arrays
    gt_values_n = np.array(gt_values)
    predicted_values = np.array(predicted_values)
    
    # Calculate the linear least squares regression line
    A = np.vstack([gt_values_n, np.ones(len(gt_values_n))]).T
    m, c = np.linalg.lstsq(A, predicted_values, rcond=None)[0]
    
    return m, c, gt_values_n

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_img(gt_values, predicted_values,flag):
    fig = plt.figure()
    # 绘制散点图
    m,c, gt_values_n = calculate_linear_least_squares(gt_values, predicted_values)
    print(m,c)
    regression_line = m * gt_values_n + c 
    plt.plot(gt_values, regression_line, color='#e35f62', label='Linear Regression')

# Add grid and legend
    plt.scatter(gt_values, predicted_values,color='limegreen',label='Estimation')
    plt.plot([10, 60], [10, 60],'k--' ,label='Ground Truth',alpha=0.6)  # 画对角线
    # plt.plot([10, 100], [10, 100],'k--' ,label='Ground Truth',alpha=0.6)  # 画对角线
    # plt.plot([10, 50], [10, 50],'k--' ,label='Ground Truth',alpha=0.6)  # 画对角线
    # 设置图形标题和坐标轴标签
    plt.title(f'Estimation with {flag}')
    plt.xlabel('Ground Truth')
    plt.ylabel('Camera estimation')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{save_dir}{flag}.png',dpi=300)
    plt.clf()
    # 显示图形
    # plt.show()

if __name__ =='__main__':
    flag_list = ['roll 0 degree','roll 5 degree','roll -5 degree','look_up','look_down','yaw_right','yaw_left']
    bias_list = [0,2,3,1,1,0,0]
    FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 
    save_dir = f'./results/measures/{FOLDER_NAME}/'
    mkdir(save_dir)
    for id, flag in enumerate(flag_list):
        with open(f'{save_dir}{flag}.csv', "w") as f1:
            csv_writer = csv.writer(f1)
            csv_writer.writerow([
                'gt', 'estimation'
                                ])

        gt_values, predicted_values = plot_rand(flag,bias_list[id])
        plot_img(gt_values, predicted_values,flag)