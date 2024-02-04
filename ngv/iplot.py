from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.optimize import leastsq
from lib2to3.refactor import MultiprocessRefactoringTool
import os
import csv
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def down_list_f():
    down_plot_gt = [               
                    15.63,
                    14.06,
                    13.55,
                    14.55,
                    18.16,
                    16.95
    ]
    down_plot_single = [
                    18.37,
                    16.06,
                    15.7,
                    16.79,
                    22.32,
                    20.23
    ]
    down_plot_yolo = [
                    17.12,
                    20.89,
                    17.29,
                    13.27,
                    21.36,
                    15.48  
    ]
    down_plot_mc = [
                    15.93,
                    15.43,
                    14.3,
                    14.29,
                    18.8,
                    16.66
    ]

    ratio = [
    [-0.17530390275112,    -0.142247510668563,    -0.158671586715867,       -0.153951890034364,    -0.229074889867841,    -0.193510324483776],
    [-0.09532949456174,    -0.485775248933144,    -0.276014760147601,       0.087972508591065,    -0.176211453744493,    0.086725663716814],
    [-0.019193857965451,    -0.097439544807966,    -0.055350553505535,       0.01786941580756,    -0.035242290748899,    0.017109144542773],
    ]


    down_list = [
    down_plot_gt,
    down_plot_single,
    down_plot_yolo,
    down_plot_mc
    ]
    return ratio,down_list

def straight_list_f():

    straight_plot_gt = [               
                   17.34,
26.39,
23.96,
16.01,
16.51

    ]
    straight_plot_single = [
                    17.34,
26.39,
23.96,
16.01,
16.51
    ]
    straight_plot_yolo = [
                    17.13,
31.19,
25.22,
15.38,
15.36
    ]
    straight_plot_mc = [
                    17.3,
27.35,
24.21,
15.88,
16.28
    ]

    ratio = [
    [0.,    0.,    0.,    0.,    0.],
    [0.012110726643599,    -0.181887078438803,    -0.052587646076795,    0.039350405996252,    0.069654754694125],
    [0.002306805074971,    -0.036377415687761,    -0.010434056761269,    0.008119925046846,    0.013930950938825]
    ]

    straight_list = [
    straight_plot_gt,
    straight_plot_single,
    straight_plot_yolo,
    straight_plot_mc
    ]
    return ratio,straight_list

def up_list_f():

    up_plot_gt = [               
                    13.85,
                    20.68,
                    26.77,
                    17.87
    ]
    up_plot_single = [
                        12.12,
                        17.3,
                        21.03,
                        15.61
    ]
    up_plot_yolo = [
                    21.04, 
                    20.9,   
                    31.24,  
                    13.56   
    ]
    up_plot_mc = [
                    15.29,
                    20.72,
                    27.66,
                    17.01
    ]

    ratio = [
    [0.124909747292419,    0.163442940038685,    0.214419125887187,    0.1264689423615],
    [-0.51913357400722,    -0.01063829787234,    -0.166977960403437,    0.241186345831002],
    [-0.103971119133574,    -0.001934235976789,    -0.033246171087038,    0.048125349748181]
    ]
    up_list = [
    up_plot_gt,
    up_plot_single,
    up_plot_yolo,
    up_plot_mc
    ]
    return ratio,up_list

def plot(cam_pose, imu_pose,tag):
    plt.scatter(cam_pose,imu_pose, s=20, c='green',label='pitch(degree) (number of value : {})'.format(len(imu_pose)))

    x = np.array(cam_pose)
    y = np.array(imu_pose)
    x_bar = x.mean()
    y_bar = y.mean()
    calculated_weight = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar)**2).sum()
    calculated_bias = y_bar - calculated_weight * x_bar

    imu_start = calculated_weight * min(cam_pose) + calculated_bias
    imu_end = calculated_weight * max(cam_pose) + calculated_bias
    plt.scatter(cam_pose,imu_pose,c='green')
    plt.plot([min(cam_pose),max(cam_pose)],[imu_start,imu_end],c='red', label='trend line (Least Square Method)')

    if tag.split('_')[-1] == 'RATIO':
        plt.plot([10,35],[-1,1], c='black', linestyle='dashed')
    else: 
        plt.plot([10,35],[10,35], c='black', linestyle='dashed')
    title_font = {
    'fontsize': 15,
    'fontweight': 'bold'
    }
    plt.legend(fontsize = '15')
    plt.title('{0}\n'.format("Look " + tag), fontdict=title_font)
    plt.xlabel('GT (m)', fontdict=title_font)
    plt.ylabel('Prediction (m)', fontdict=title_font)
    ax=plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.grid(True)
    plt.show()

def sort_lists(sort_list,samp_list):
    ind_sort = []
    temp = sorted(sort_list)
    for i in sort_list:
        ind = temp.index(i)
        ind_sort.append(ind)
    temp_list =[]
    for o in ind_sort:
        temp_list.append(samp_list[o])
    return temp_list

def Fun(p,x):                        # 定义拟合函数形式
    a1,a2,a3 = p
    return a1*x**2+a2*x+a3

def error (p,x,y):                    # 拟合残差
    return Fun(p,x)-y 

def iplot(samp_ratio,samp_list,tag):
#############'hotpink'#88c999
    gt_ratio = samp_list[0]
    single_ratio = samp_ratio[0]
    yolo_ratio = samp_ratio[1]
    mc_ratio = samp_ratio[2]

    single_ratio = sorted(single_ratio)
    yolo_ratio = sorted(yolo_ratio)
    mc_ratio = sorted(mc_ratio)

    plot(gt_ratio,single_ratio,tag+ '_RATIO')
    plot(gt_ratio,yolo_ratio,tag+ '_RATIO')
    plot(gt_ratio,mc_ratio,tag+ '_RATIO')
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111) 
    
    # ax1.scatter(gt_ratio,single_ratio,color='lightblue',label='single')
    # ax1.scatter(gt_ratio,yolo_ratio,color='darkblue',label='yolo')
    # ax1.scatter(gt_ratio,mc_ratio,color='hotpink',label='mc')
    
    # ax1.set_xlabel("Distance (m)")
    # ax1.set_ylabel("Distance (m)") 

    # ax1.set_xlim(10,35)   
    # ax1.set_ylim(-0.8,0.5)  
    # plt.suptitle("Look " + tag + '_RATIO')
    # plt.legend()
    # plt.grid()

    gt = samp_list[0]
    gt_s = [10,40]
    gt_0 = [10,10,10,10,10]
    single = samp_list[1]
    yolo = samp_list[2]
    mc = samp_list[3]

    # x = np.linspace(10,40,100)  # 创建时间序列
    # p_value = [-2,5,10] # 原始数据的参数
    # y = Fun(p_value,x)
    # p0 = [0.1,-0.01,100] # 拟合的初始参数设置
    # print(p0)
    # para =leastsq(error, p0, args=(x,y)) # 进行拟合
    # y_fitted = Fun (para[0],x) # 画出拟合后的曲线
    
    single = sort_lists(gt,single)
    yolo = sort_lists(gt,yolo)
    mc = sort_lists(gt,mc)

    plot(gt,single,tag + '_DISTANCE')
    plot(gt,yolo,tag + '_DISTANCE')
    plot(gt,mc,tag + '_DISTANCE')

    # single = sorted(single)
    # yolo = sorted(yolo)
    # mc = sorted(mc)

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111) 
    # ax2.plot(gt_s,gt_s,color='red',label='gt')
    
    # ax2.scatter(single,gt,color='lightblue',label='single')
    # ax2.scatter(yolo,gt,color='darkblue',label='yolo')
    # ax2.scatter(mc,gt,color='hotpink',label='mc')

    # ax2.set_xlabel("Distance (m)")
    # ax2.set_ylabel("Distance (m)") 

    # ax2.set_xlim(10,40)   
    # ax2.set_ylim(10,40)  

    # # plt.savefig(title+'.jpg', dpi=1080) #可以存到本地，高清大图。路径默认为当前路径，dpi可理解为清晰度
    # plt.suptitle("Look " + tag + '_DISTANCE')
    # plt.legend()
    # plt.grid()
    # plt.show()


if __name__=="__main__":
    up_ratio,up_list = up_list_f()
    straight_ratio,straight_list = straight_list_f()
    down_ratio,down_list = down_list_f()
    iplot(up_ratio,up_list,'Up')
    iplot(straight_ratio,straight_list,'straight')
    iplot(down_ratio,down_list,'Down')
