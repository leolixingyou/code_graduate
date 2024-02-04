import os
import time
import numpy as np
import copy
import cv2

from quternian2euler import quaternion_to_euler

import rospy
from sensor_msgs.msg import CompressedImage
from tf2_msgs.msg import TFMessage

FOLDER_NAME = time.strftime("%Y_%m_%d", time.localtime()) 
SAVE_DIR = './dataset/' + FOLDER_NAME + '/'

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

class Calib_Node:
    def __init__(self):
        rospy.init_node('Calib_node')

        self.get_f60_new_image = False

        self.Camera_f60_bbox = None
        self.tf_data = None
        self.cur_f60_img = {'img':None, 'header':None}
        self.tf_check_list = []
        
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.IMG_f60_callback)
        rospy.Subscriber('/tf', TFMessage, self.tf_callback)

    def IMG_f60_callback(self,msg):
        t1= time.time()
        if not self.get_f60_new_image:
            # print('======== IMAGE ========')
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True
            # print('IMAGE_Sub is :', round(1/(time.time() - t1),2),' FPS')

    def tf_callback(self,msg):
        t1= time.time()
        # print('======== TF ========')
        child_id = msg.transforms[1].child_frame_id
        roat = msg.transforms[1].transform.rotation
        if child_id == 'Camera':
            # quaternion_list = [roat.x, roat.y, roat.z, roat.w]
            quaternion_list = [roat.w, roat.x, roat.y, roat.z]
            self.tf_data = [round(x,0) for x in quaternion_to_euler(quaternion_list)]
            # print('TF_Sub is :', round(1/(time.time() - t1),2),' FPS')
    
    def main(self):
        while not rospy.is_shutdown():
            if self.tf_data != None:
                tf_data = copy.copy(self.tf_data)
                orig_im_f60 = copy.copy(self.cur_f60_img['img'])
                print('tf_data is :',self.tf_data)
                self.get_f60_new_image=False
                
if __name__ == "__main__":
    Calib_Node = Calib_Node()
    Calib_Node.main()

