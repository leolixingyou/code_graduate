from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import random
import time
import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from std_msgs.msg import Float64MultiArray 
import signal, sys
import math

class plot_distance():
    def __init__(self):
        self.est_In_Signal = False
        self.gt_In_Signal  = False

        self.ct1 = 0
        self.ct2 = 0

        rospy.init_node('Pitch_plot')
        rospy.Subscriber('/est_euler', Float64MultiArray, self.est_CallBack )
        rospy.Subscriber('/gt_euler', Float64MultiArray, self.gt_CallBack)
        
        fig = plt.figure()   
        
        ax = plt.subplot(111, xlim=(0, 50), ylim=(-0.5, 0.5))
        ax.set_title('Pitch')
        ax.set(xlabel='time (0.1 s)', ylabel='euler( degree )')

        self.Est_pitch, = ax.plot(np.arange(50), np.ones(50, dtype=np.float)*np.nan, lw=1, c='blue',ms=1, label='Est(Camera)')
        self.GT_pitch, = ax.plot(np.arange(50), np.ones(50, dtype=np.float)*np.nan, lw=1, c='red',ms=1, label='GT(IMU)')
        ax.legend(loc='upper right', fontsize=8,shadow=True)

        anim = animation.FuncAnimation(fig, self.animate, init_func= self.init ,frames=200, interval=50, blit=False)
        
        fig.tight_layout()
        plt.show()

    def est_CallBack(self, msg):
        if not self.est_In_Signal:
            self.est_arr = []
            for est in msg.data:
                self.est_arr.append(est)
            self.est_In_Signal = True
            self.ct1+=1
           
    def gt_CallBack(self, msg):
        if not self.gt_In_Signal:
            self.gt_arr = []
            for gt in msg.data:
                self.gt_arr.append(gt)
            self.gt_In_Signal = True
            self.ct2+=1

    def init(self): return self.Est_pitch, self.GT_pitch

    def animate(self, i):
        if self.gt_In_Signal and self.est_In_Signal: 
            print(self.ct1, self.ct2)
            y = self.est_arr[1]
            y2 = self.gt_arr[1]

            old_y = self.Est_pitch.get_ydata()
            old_y2 = self.GT_pitch.get_ydata()

            new_y = np.r_[old_y[1:], y]
            new_y2 = np.r_[old_y2[1:], y2]

            self.Est_pitch.set_ydata(new_y)
            self.GT_pitch.set_ydata(new_y2)

            self.est_In_Signal = False
            self.gt_In_Signal = False

            return self.Est_pitch, self.GT_pitch    


def signal_handler(signal, frame):
        print("Exit")
        exit(0)

if __name__=="__main__":
    signal.signal(signal.SIGINT, signal_handler)
    pl_dist = plot_distance ()
 
    