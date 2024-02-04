
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from intrinsic.calibration import Calibration

class CameraGeo(object):
    def __init__(self, height=1.3, yaw_deg=0, pitch_deg=-5, roll_deg=0, image_width=1024, image_height=512, field_of_view_deg=60,intrinsic_matrix=None):
        # scalar constants
        self.height = height
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.yaw_deg = yaw_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg
        # camera intriniscs and extrinsics
        self.intrinsic_matrix = intrinsic_matrix
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        
        ratio = [(1920/1280),(1080/720)]
        self.intrinsic_matrix_1280 = [
                                        [intrinsic_matrix[0][0]*ratio[0], intrinsic_matrix[0][1]         , intrinsic_matrix[0][2]*ratio[0]],
                                        [intrinsic_matrix[1][0]         , intrinsic_matrix[1][1]*ratio[1], intrinsic_matrix[1][2]*ratio[1]],
                                        [intrinsic_matrix[2][0]         , intrinsic_matrix[2][1]         , intrinsic_matrix[2][2]         ]
                                     
                                        ]
        self.inverse_intrinsic_matrix_1280 = np.linalg.inv(self.intrinsic_matrix_1280)

        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                            [cp*sr, cp*cr, sp],
                                            [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
        self.rotation_cam_to_road = rotation_road_to_cam.T # for rotation matrices, taking the transpose is the same as inversion
        self.translation_cam_to_road = np.array([0,-self.height,0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3,0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3,3] = self.translation_cam_to_road
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])
        

        combination_r = [roll,pitch,yaw]
        new_r = self.make_rotation(combination_r)
        self.rt = np.array([
                                [new_r[0][0],new_r[0][1],new_r[0][2],0],
                                [new_r[1][0],new_r[1][1],new_r[1][2],0],
                                [new_r[2][0],new_r[2][1],new_r[2][2],-height],
                                ])
        
        # self.KRT = np.dot(intrinsic_matrix, self.rt)
        # print(f'KRT is {self.KRT}')
        
        # ### ioniq5 with 우붕이
        # range_3d = [100, 3, 6]
        # ###look straight
        # src_poi =np.array([
        #             [571,404],
        #             [289,718],
        #             [1035,718],
        #             [615,404]
        #             ])

        ## ioniq5 with niro
        ##look straight
        range_3d = [80, 3, 6]
        src_poi =np.array([
                    [571,404],
                    [289,718],
                    [1035,718],
                    [615,404]
                    ])

        # ##look down
        # range_3d = [80, 3, 6]
        # src_poi =np.array([
        #             # [507,391],
        #             [562,354],
        #             [5,688],
        #             [1273,688],
        #             [688,361]
        #             ])

        # # ###look up
        # range_3d = [80, 3, 6]
        # src_poi =np.array([
        #             [572,443],
        #             [97,718],
        #             [1213,720],
        #             [685,443]
        #             ])




        # ### morai
        # range_3d = [43, 3, 8]
        # src_poi =np.array([
        #             [599,395],
        #             [289,718],
        #             [1035,718],
        #             [681,394]
        #             ])

        self.ground_3d = np.array([
                        [range_3d[0], -range_3d[1],    0],
                        [range_3d[2], -range_3d[1],    0],
                        [range_3d[2],  range_3d[1],    0],
                        [range_3d[0],  range_3d[1],    0]

                        # [range_3d[0]    , 0             , 0             ,range_3d[0]],
                        # [-range_3d[1]   , -range_3d[1]  , range_3d[1]   ,range_3d[1]],
                        # [0              ,  0            , 0             ,0  ],
                        # [1              ,  1            , 1             ,1  ]
                        ])
        self.KRT,self.invKRT = self.get_M_matrix(src_poi)
        
        self.cal_angle()
        self.src_poi = src_poi
        
        # yaw_initial = -0.6810110838532153
        # pitch_initial = -0.21769690844034348
        # roll_initial = 0.0
    
    def update_angle(self,yaw_deg,pitch_deg):
        init_pose = [0,self.pitch_deg,0]
        roll_ini_degree,pitch_ini_degree,yaw_ini_degree = self.euler2degrees(init_pose)

        if yaw_ini_degree - self.yaw_deg > 0.1:
            cur_yaw = math.radians(yaw_deg-yaw_ini_degree)
            cur_pitch = math.radians(pitch_deg-pitch_ini_degree)

            cur_pose = [0,cur_pitch,0] 
            self.KRT,self.src_poi =self.getRotationMatrix(cur_pose)        

    def euler2degrees(self, cur_pose):
        roll = math.degrees(cur_pose[0])
        pitch = math.degrees(cur_pose[1])
        yaw = math.degrees(cur_pose[2])

        return roll, pitch, yaw

    def getNewGroundPlane(self, new_R):
        pts_3d =self.ground_3d
        # pts_leftline_3d = copy.deepcopy(self.calib.pts_leftline_3d_init)
        # pts_rightline_3d = copy.deepcopy(self.calib.pts_rightline_3d_init)

        for i, pt_3d in enumerate(pts_3d):
            pts_3d[i] = np.dot(new_R, pt_3d.transpose())
        new_pts_2d = self.project_3d_to_2d(pts_3d)
        print(f'new_pts_2d is {new_pts_2d}')
        new_src_pt = np.float32([[int(np.round(new_pts_2d[0][0])), int(np.round(new_pts_2d[0][1]))],
                                [int(np.round(new_pts_2d[1][0])), int(np.round(new_pts_2d[1][1]))],
                                [int(np.round(new_pts_2d[2][0])), int(np.round(new_pts_2d[2][1]))],
                                [int(np.round(new_pts_2d[3][0])), int(np.round(new_pts_2d[3][1]))]])

        new_M2 = cv2.getPerspectiveTransform(new_src_pt, self.dst_pt)
        ground_plane = np.array(new_src_pt, np.int32)
        # return new_M2, ground_plane, left_line, right_line
        return new_M2, ground_plane
        
    def getRotationMatrix(self, cur_pose):
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
        

        return self.getNewGroundPlane(new_R)

    def cal_angle(self):
        rotation = np.dot(self.KRT,self.inverse_intrinsic_matrix_1280)
        ro = R.from_matrix(rotation)
        angles = ro.as_euler('zyx',degrees='True')

    def distance_cal(self,points,height):
        points = np.array([
                            [points[0][0]],
                            [points[0][1]],
                            [height]
                            ])
        pts_3d = np.dot(self.invKRT, points)

        pts_3d = pts_3d / pts_3d[2,:]
        pts_3d = pts_3d.T[:,:2]
        dist = np.sqrt(pts_3d[0][0]**2 + pts_3d[0][1]**2)
        return [dist]

    def project_3d_to_2d(self, points):
        
        points = np.dot(self.KRT, points.transpose())
        points[:2, :] /= points[2, :]
        src_pt = np.float32([[int(np.round(points[0][0])), int(np.round(points[1][0]))],
                             [int(np.round(points[0][1])), int(np.round(points[1][1]))],
                             [int(np.round(points[0][2])), int(np.round(points[1][2]))],
                             [int(np.round(points[0][3])), int(np.round(points[1][3]))]])
        return src_pt

    def make_rotation(self,cur_pose):
            ### cur_pose radian: roll pitch yaw 
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
            return new_R

    def get_M_matrix(self, pts_2d):
        src_pt = np.float32([[int(np.round(pts_2d[0][0])), int(np.round(pts_2d[0][1]))],
                             [int(np.round(pts_2d[1][0])), int(np.round(pts_2d[1][1]))],
                             [int(np.round(pts_2d[2][0])), int(np.round(pts_2d[2][1]))],
                             [int(np.round(pts_2d[3][0])), int(np.round(pts_2d[3][1]))]])

        self.dst_pt = np.float32([[self.ground_3d[0][0], self.ground_3d[0][1]], 
                                [self.ground_3d[1][0], self.ground_3d[1][1]], 
                                [self.ground_3d[2][0], self.ground_3d[2][1]], 
                                [self.ground_3d[3][0], self.ground_3d[3][1]]])

        # M2 = cv2.getPerspectiveTransform(src_pt, self.dst_pt)
        M2,_ = cv2.findHomography(src_pt, self.dst_pt)
        M1,_ = cv2.findHomography(self.dst_pt, src_pt)
        
        return M1, M2



    def camframe_to_roadframe(self,vec_in_cam_frame):
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_camframe(self,u,v):
        # NOTE: The results depend very much on the pitch angle (0.5 degree error yields bad result)
        # Here is a paper on vehicle pitch estimation:
        # https://refubium.fu-berlin.de/handle/fub188/26792
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height*Kinv_uv_hom/denominator
    
    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y]) # read book section on coordinate systems to understand this

    def compute_minimum_v(self, dist):
        """
        Find cut_v such that pixels with v<cut_v are irrelevant for polynomial fitting.
        Everything that is further than `dist` along the road is considered irrelevant.
        """        
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v
if __name__=='__main__':
    calib = ['./intrinsic/f60.txt','./intrinsic/f_camera_i30.txt']
    vehicle_height = 1.85
    obj_height = [vehicle_height + 4.4][0]
    obj_height = 1.85

    yaw = 1.1967836642392118
    pitch = -3.2907187963631257
    roll = 97.71791594937976

    #### #hyper param
    img_size= [[1920,1080],[1280,720]]
    # img_size= [[1920,1080],[1280,806]]
    cal = Calibration(calib,img_size)

    geo = CameraGeo(height = obj_height, yaw_deg = yaw, pitch_deg = pitch, roll_deg = roll, intrinsic_matrix = cal.camera_matrix_f60)
