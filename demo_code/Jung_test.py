import cv2
import numpy as np
from Jung_method import MotionEstimation
from Jung_method_beta import MotionEstimation_beta
from tools import *

class MOTION_TEST:
    def __init__(self):
        pass
    
    def img_process(self,img_dir,estimator,init_img):
        for img_file in img_dir:
            gt_from_name = img_file.split('/')[-1].split('.png')[0]
            img = cv2.imread(img_file, 0)
            pitch, yaw, roll = estimator.main(img, init_img)
            print(f"GT is : {gt_from_name}")
            print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

    def main(self):
        img_dir = '../dataset/2023_04_23/'
        estimator = MotionEstimation()
        estimator_beta = MotionEstimation_beta()
        img_list =sorted(get_image_list(img_dir))

        init_img = cv2.imread('../dataset/2023_04_23/pith_yaw0/pitch_0.0_yaw_0.0.png', 0)
        self.img_process(img_list,estimator_beta,init_img)

if __name__ == "__main__":
    motion_test = MOTION_TEST()
    motion_test.main()

