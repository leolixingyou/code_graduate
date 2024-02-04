import os
import cv2
import numpy as np

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

        self.f60_intrinsic = np.array([[cam_param_f60[0]*(tar_size[0]/raw_size[0]), cam_param_f60[1], cam_param_f60[2]*(tar_size[0]/raw_size[0])], 
                                       [cam_param_f60[3], cam_param_f60[4]*(tar_size[1]/raw_size[1]), cam_param_f60[5]*(tar_size[1]/raw_size[1])], 
                                       [cam_param_f60[6], cam_param_f60[7], cam_param_f60[8]]])

        print('f60_intrinsic',self.f60_intrinsic)


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
                './calibration_data/f60.txt',
                './calibration_data/f120.txt',
                './calibration_data/r120.txt'
                ]
    cal = Calibration(camera_path)
    front_img = cv2.undistort(front_img, self.calib.camera_matrix_f60, self.calib.dist_coeffs__f60, None, self.calib.camera_matrix_f60)
