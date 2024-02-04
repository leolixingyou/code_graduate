from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import random
import time
import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray 
import signal, sys
import math

class plot_distance():
    def __init__(self):
        self.pre_error = 0.0
        self.total_error = []
        self.dist_arr = []
        self.GT = []
        self.new_dist_CallBack_flag = False
        self.new_gt_CallBack_flag = False

        rospy.init_node('disance_plot')
        rospy.Subscriber('/distance_arr', Float64MultiArray, self.dist_CallBack, queue_size=1)
        rospy.Subscriber('/lidar/postpoint', MarkerArray, self.LiDARcallback, queue_size=10)

        fig = plt.figure()   
        
        ax = plt.subplot(211, xlim=(0, 50), ylim=(0, 50))
        ax.set_title('GT(LiDAR) and Estimate Distance(Camera)')
        ax.set(xlabel='time (0.1 s)', ylabel='distance( m )')

        ax_2 = plt.subplot(212, xlim=(0, 50), ylim=(-3, 20))
        ax_2.set_title('Error (GT(LiDAR) - Distance(Camera))')
        ax_2.set(xlabel='time (0.1 s)', ylabel='distance( m )')
        
        self.line1, = ax.plot(np.arange(50), np.ones(50, dtype=np.float)*np.nan, lw=1, c='blue',ms=1, label='GT(LiDAR)')
        self.line1_2, = ax.plot(np.arange(50), np.ones(50, dtype=np.float)*np.nan, lw=1, c='red',ms=1, label="Distance(Camera)")
        ax.legend(loc='upper right', fontsize=8,shadow=True)

        self.line2, = ax_2.plot(np.arange(50), np.ones(50, dtype=np.float)*np.nan, lw=1,ms=1, label='Error')
        ax_2.legend(loc='upper right', fontsize=8,shadow=True)

        anim = animation.FuncAnimation(fig, self.animate, init_func= self.init ,frames=200, interval=50, blit=False)
        anim_2 = animation.FuncAnimation(fig, self.animate_2  , init_func= self.init_2 ,frames=200, interval=50, blit=False)

        fig.tight_layout()
        plt.show()
        rospy.spin()


    def dist_CallBack(self, msg):
        self.dist_arr = []
        for dist in msg.data:
            self.dist_arr.append(round(dist, 2))  
        
        self.new_dist_CallBack_flag = True

    def LiDARcallback(self, msg):
        self.GT = []
        if len(msg.markers) > 0:
            for marker in msg.markers:
                lidar_x = -marker.pose.position.x
                lidar_y = marker.pose.position.y
                lidar_z = marker.pose.position.z
        
                GT_dist = math.sqrt(lidar_x**2+lidar_z**2)
            self.GT.append(GT_dist)
        
    def init(self):
        return self.line1, self.line1_2
        
    def init_2(self):
        return self.line2

    def animate(self, i):
        if (len(self.dist_arr) > 0 and len(self.GT) > 0) and (len(self.dist_arr)==len(self.GT)):
            y = self.GT
            y2 = self.dist_arr
        else:
            y = None
            y2 = None

        old_y = self.line1.get_ydata()
        new_y = np.r_[old_y[1:], y]

        old_y2 = self.line1_2.get_ydata()
        new_y2 = np.r_[old_y2[1:], y2]

        self.line1.set_ydata(new_y)
        self.line1_2.set_ydata(new_y2)

        return self.line1, self.line1_2

    def animate_2(self, i):
        Error_list = []
        if self.new_dist_CallBack_flag:
            if (len(self.dist_arr) > 0 and len(self.GT) > 0) and len(self.dist_arr)==len(self.GT):
                for gt, dist in zip(self.GT, self.dist_arr):
                    error = abs(gt - dist)
                    print(gt, dist, error)
                    Error_list.append(error)
                    if self.pre_error != (error/gt)*100:
                        self.total_error.append((error/gt)*100)
                        self.pre_error = (error/gt)*100
                mean_error = sum(Error_list)/len(Error_list)
                y_2 = mean_error
                
            else:
                y_2 = 0
                
            old_y_2 = self.line2.get_ydata()
            new_y_2 = np.r_[old_y_2[1:], y_2]
            self.line2.set_ydata(new_y_2)

            self.new_dist_CallBack_flag = False
            return self.line2
        
        else: pass


def signal_handler(signal, frame):
        print("Exit")
        exit(0)

if __name__=="__main__":
    signal.signal(signal.SIGINT, signal_handler)
    pl_dist = plot_distance()
   