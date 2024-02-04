import numpy as np
import math
import cv2
import pose
import copy
from eulerangles import euler2matrix, matrix2euler

class Distance:
    def __init__(self, calibration):
        self.calib = calibration

        self.center_pts = []
        self.past_dist = 0

        self.pitch_threshold = 0.5 #degrees
        self.adaptive_flag = False
    
    def euler2degrees(self, cur_pose):
        roll = math.degrees(cur_pose[0])
        pitch = math.degrees(cur_pose[1])
        yaw = math.degrees(cur_pose[2])

        return roll, pitch, yaw

    def getDistance(self, bboxes, cur_pose, task):
        send_LiDAR_pt = []
        detect_pts = []
        dist_arr = []
        ori_dist_arr = []

        self.center_pts = []

        self.Euler = []
        self.translation_Z = 0
        self.init_translation_z = 1.6152
        
        self.get_new_img_msg = False

        roll, pitch, yaw = self.euler2degrees(cur_pose)
        if self.pitch_threshold < abs(pitch):
            self.adaptive_flag = True

        for idx, bbox in enumerate(bboxes):
            left_top_x = bbox[0]
            left_top_y = bbox[1]
            right_bot_x = bbox[2]
            right_bot_y = bbox[3]
            
            ## center of bottom
            center_pt = [left_top_x + ((right_bot_x -left_top_x)/2), right_bot_y]
            self.center_pts.append(center_pt)
            detect_pts.append(np.float32([[center_pt[0]], [center_pt[1]], [1.0]]))
        if task == "front":
            ori_pts_3d = np.dot(self.calib.M, detect_pts)

        elif task == "left": 
            ori_pts_3d = np.dot(self.calib.LM, detect_pts)

        elif task == "right": 
            ori_pts_3d = np.dot(self.calib.RM, detect_pts)
    
        ori_pts_3d = np.squeeze(ori_pts_3d, axis=2)
        ori_pts_3d = ori_pts_3d / ori_pts_3d[2,:]
        ori_pts_3d = ori_pts_3d.T[:,:2]

        if self.adaptive_flag:
            # new_M2, ground_plane, left_line, right_line = self.getRotationMatrix(cur_pose, task)
            new_M2, ground_plane = self.getRotationMatrix(cur_pose, task)

            if task == "front":
                pts_3d = np.dot(new_M2, detect_pts)

            elif task == "left": 
                # pts_3d = np.dot(np.dot(new_M2, self.calib.LM), detect_pts)
                pts_3d = np.dot(new_M2, detect_pts)

            elif task == "right": 
                pts_3d = np.dot(new_M2, detect_pts)

            pts_3d = np.squeeze(pts_3d, axis=2)
            pts_3d = pts_3d / pts_3d[2,:]
            pts_3d = pts_3d.T[:,:2]

            new_dist_flag = True
            self.adaptive_flag = False

        else:
            if task == "front":
                pts_3d = np.dot(self.calib.M, detect_pts)
                ground_plane = np.array(self.calib.src_pt, np.int32)
                # left_line = self.calib.src_leftline_2d
                # right_line = self.calib.src_rightline_2d

            elif task == "left": 
                pts_3d = np.dot(self.calib.LM, detect_pts)
                ground_plane = np.array(self.calib.l_src_pt, np.int32)
            elif task == "right": 
                pts_3d = np.dot(self.calib.RM, detect_pts)
                ground_plane = np.array(self.calib.r_src_pt, np.int32)

            pts_3d = np.squeeze(pts_3d, axis=2)
            pts_3d = pts_3d / pts_3d[2,:]
            pts_3d = pts_3d.T[:,:2]
            new_dist_flag = False
            
        for pt_3d in pts_3d:
            if abs(pt_3d[1]) > 1: continue
            dist = np.sqrt(pt_3d[0]**2 + pt_3d[1]**2)
            if dist > 4 :
                dist_arr.append([round(dist, 2)])
                np.diff(dist_arr)
                send_LiDAR_pt.append(pt_3d)

        # ##temp
        for ori_pt_3d in ori_pts_3d:
            if abs(ori_pt_3d[1]) > 1: continue
            ori_dist = np.sqrt(ori_pt_3d[0]**2 + ori_pt_3d[1]**2)
            ori_dist_arr.append([round(ori_dist, 2)])
            # send_LiDAR_pt.append(ori_pt_3d)
  
        # return ori_dist_arr,dist_arr, send_LiDAR_pt, ground_plane, new_dist_flag, left_line, right_line
        return ori_dist_arr,dist_arr, send_LiDAR_pt, ground_plane, new_dist_flag
        
    def getNewGroundPlane(self, new_R, task):
        pts_3d = np.array([self.calib.pt1_2, self.calib.pt2_2, self.calib.pt3_2, self.calib.pt4_2])
        # pts_leftline_3d = copy.deepcopy(self.calib.pts_leftline_3d_init)
        # pts_rightline_3d = copy.deepcopy(self.calib.pts_rightline_3d_init)

        for i, pt_3d in enumerate(pts_3d):
            pts_3d[i] = np.dot(new_R, pt_3d.transpose())
        new_pts_2d = self.calib.project_3d_to_2d(pts_3d.transpose(), task)
        new_src_pt = np.float32([[int(np.round(new_pts_2d[0][0])), int(np.round(new_pts_2d[1][0]))],
                                [int(np.round(new_pts_2d[0][1])), int(np.round(new_pts_2d[1][1]))],
                                [int(np.round(new_pts_2d[0][2])), int(np.round(new_pts_2d[1][2]))],
                                [int(np.round(new_pts_2d[0][3])), int(np.round(new_pts_2d[1][3]))]])

        new_M2 = cv2.getPerspectiveTransform(new_src_pt, self.calib.dst_pt)
        ground_plane = np.array(new_src_pt, np.int32)
        # return new_M2, ground_plane, left_line, right_line
        return new_M2, ground_plane
        
    def getRotationMatrix(self, cur_pose, task):
        mode =1
        if mode == 1 :
            # R_z
            mat_yaw = np.array([[math.cos(cur_pose[2]), -math.sin(cur_pose[2]), 0],
                                [math.sin(cur_pose[2]), math.cos(cur_pose[2]), 0],
                                [0, 0, 1]]) 
            ## R_y
            mat_pitch = np.array([[math.cos(cur_pose[1]),0,math.sin(cur_pose[1])],
                                [0,1,0],
                                [-math.sin(cur_pose[1]),0,math.cos(cur_pose[1])]])
            ## R_x
            mat_roll = np.array([[1, 0, 0],
                                [0, math.cos(cur_pose[0]), -math.sin(cur_pose[0])],
                                [0, math.sin(cur_pose[0]), math.cos(cur_pose[0])]])

            new_R = np.dot(mat_yaw, np.dot(mat_pitch, mat_roll))
        

        return self.getNewGroundPlane(new_R, task)