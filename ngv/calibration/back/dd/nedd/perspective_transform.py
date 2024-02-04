import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def txt_read(LiDAR_cam_path):
    lidar_calib_param = []
    with open(LiDAR_cam_path, 'r') as f:
        for line in f.readlines():
            lidar_calib_param.append([float(i) for i in line.split(',')])
    return lidar_calib_param
        
def write_txt(output_path,txt_name,M):
    with open(output_path + txt_name, 'w') as f:
        for t,i in enumerate(M):
            if t == len(M)-1:
                f.write(' %s,%s,%s'%(i[0], i[1], i[2]))
            else:
                f.write(' %s,%s,%s,'%(i[0], i[1], i[2]))


def perspective_transform(path,output_path,txt_name):

    #left 
    l_path = path[0]
    le_txt_name = txt_name[0]
    src = np.float32(txt_read(l_path[0]))
    dst = np.float32(txt_read(l_path[1]))

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    
    write_txt(output_path,le_txt_name,m)
    
    r_path = path[1]
    ri_txt_name = txt_name[1]
    src = np.float32(txt_read(r_path[0]))
    dst = np.float32(txt_read(r_path[1]))

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    write_txt(output_path,ri_txt_name,m)

if __name__ == '__main__':
    ri_front_path = './front_ri.txt'
    right_path = './right.txt'

    le_front_path = './front_le.txt'
    left_path = './left.txt'
    
    output_path = './'
    txt_name_r = 'ri_result.txt'
    txt_name_l = 'le_result.txt'
    txt_name = [txt_name_l,txt_name_r]
    path = [[le_front_path,left_path],[ri_front_path,right_path]]
    perspective_transform(path,output_path,txt_name)
