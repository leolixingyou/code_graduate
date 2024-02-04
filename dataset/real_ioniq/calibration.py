import os
import cv2
import numpy as np

MAT_EXT = ['.jpg','.png']

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

def get_mat_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in MAT_EXT:
                image_names.append(apath)
    return image_names
class Calibration:
    def __init__(self, path,img_size):
        # camera parameters
        f60= path[0]
        cam_param_f60 = []
        print(path)
        print(f60)
        with open(f60, 'r') as f:
            data = f.readlines()
            for content in data:
                content_str = content.split()
                for compo in content_str:
                    cam_param_f60.append(float(compo))
        self.camera_matrix_f60 = np.array([[cam_param_f60[0], cam_param_f60[1], cam_param_f60[2]], 
                                       [cam_param_f60[3], cam_param_f60[4], cam_param_f60[5]], 
                                       [cam_param_f60[6], cam_param_f60[7], cam_param_f60[8]]])
        self.dist_coeffs__f60 = np.array([[cam_param_f60[9]], [cam_param_f60[10]], [cam_param_f60[11]], [cam_param_f60[12]]])
        print('camera_matrix_f60',self.camera_matrix_f60)
        print('dist_coeffs__f60',self.dist_coeffs__f60)
        raw_size, tar_size = img_size


    def undistort(self, img,flag):
        w,h = (img.shape[1], img.shape[0])
        ### The only one camera which is needed is FOV 190 camera
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 0)
        # result_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        if flag == 'f60':
            result_img = cv2.undistort(img, self.dist_coeffs__f60, self.dist_coeffs__f60, None, self.dist_coeffs__f60)
            return result_img

if __name__ == "__main__":

    camera_path = [
                './good_performance/f60.txt',
                ]
    cal = Calibration(camera_path,[1,1])
    file_list = get_mat_list('../distance/')
    save_dir = './out_dist/'
    make_dir(save_dir)
    for image in file_list:
        fname = image.split('/')[-1][:-4]
        img = cv2.imread(image)
        front_img = cv2.undistort(img, cal.camera_matrix_f60, cal.dist_coeffs__f60, None, cal.camera_matrix_f60)
        print(fname)
        cv2.imwrite(f'{save_dir}{fname}.jpg',front_img)