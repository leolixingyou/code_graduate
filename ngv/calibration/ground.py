
import numpy as np
import rospy
import sensor_msgs.point_cloud2
from std_msgs.msg import Float32MultiArray 
from sensor_msgs.msg import PointCloud2
from sklearn import linear_model, datasets

class Calibration:
    def __init__(self):

        rospy.init_node('ground', anonymous = True)
        rospy.Subscriber('/lidar/front_ground_cloud', PointCloud2, self.Lidar_ground_callback)
        # self.plane = Plane3D(self.ground)
        # print(self.plane)
        self.ground = 0
        self.ground_count =0
        self.ground_points = [[0,0,-1.7],
                              [0,0,-1.7],
                              [0,0,-1.7],
                              [0,0,-1.7]]
        self.count = 0

    def Lidar_ground_callback(self,msg):
        print('start')
        points = []

        for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            condition_x = (point[0] < -3) and (point[0]>-50) 
            condition_y = (point[1] > -1) and (point[1]< 1) 
            condition_z = (point[2] < -1) 
            if condition_x and condition_y and condition_z:
                temp = [-round(point[0],3),round(point[1],3),round(point[2],3)]
                points.append(temp)
        self.ground_points = points

    def main(self):
        self.ground = np.array(self.ground_points)
        if (len(self.ground) >4) and (self.count == 0):        
            self.count += 1
            self.write_txt(self.ground)

    def write_txt(self,ground):
        print('writing')
        with open("ground_points.txt", 'w') as f:
            print(len(ground))
            for i in range(len(ground)):
                for o in range(3):
                    # print(ground[i][o])
                    f.write(str(ground[i][o]))
                    if o == 2:
                        f.write('\n')
                    else:
                        f.write(', ')
    def read_txt(self):
        ground_points = []
        with open('ground_points.txt', 'r') as f:
            for line in f.readlines():
                ground_points.append([float(i) for i in line.split(',')])
        print(ground_points)
            
        # self.Left_camera_RT = np.array([[ground_points[0], ground_points[1], ground_points[2], ground_points[9]],
        #                                 [ground_points[3], ground_points[4], ground_points[5], ground_points[10]],
        #                                 [ground_points[6], ground_points[7], ground_points[8], ground_points[11]]])            

if __name__ == "__main__":
    cal = Calibration()
    cal.read_txt()
    while not rospy.is_shutdown():
        cal.main()
