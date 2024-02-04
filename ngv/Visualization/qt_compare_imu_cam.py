import sys, os, random
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import numpy as np
import math
import tf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
from sensor_msgs.msg import Imu

import rospy
import time
from std_msgs.msg import Float64MultiArray 

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        
        self.axex1 = fig.add_subplot(311, xlim=(0, 50), ylim=(-3.0, 3.0))
        self.axex1.set_title('Euler(degrees_roll)')
        self.axex1.set(xlabel='time (0.1 s)', ylabel='degree')
        
        self.axex2 = fig.add_subplot(312, xlim=(0, 50), ylim=(-10.0, 5.0))
        self.axex2.set_title('Euler(degrees_pitch)')
        self.axex2.set(xlabel='time (0.1 s)', ylabel='degree')

        self.axes3 = fig.add_subplot(313, xlim=(0, 50), ylim=(-180, 180.0))
        self.axes3.set_title('Euler(degrees_yaw)')
        # self.axex.set(xlabel='m/s', ylabel='degree')

        fig.tight_layout()
        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        
    def compute_initial_figure(self):
        pass

class AnimationWidget(QWidget):
    def __init__(self):
        QMainWindow.__init__(self)
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(self, width=7, height=8, dpi=100)
        vbox.addWidget(self.canvas)

        ## Button
        hbox = QHBoxLayout()
        self.start_button = QPushButton("start", self)
        self.stop_button = QPushButton("stop", self)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        ## plot
        self.x = np.arange(50)
        self.y = np.ones(50, dtype=np.float)*np.nan
        
        self.cam_roll_line, = self.canvas.axex1.plot(self.x, self.y, '.-', markersize=5,animated=True, label='cam_roll', color='blue',lw=1)
        self.imu_roll_line, = self.canvas.axex1.plot(self.x, self.y, '.-', markersize=5,animated=True, label='imu_roll', color='red',lw=1)
        self.canvas.axex1.legend(loc='upper right', fontsize=7, ncol=3, shadow=True, borderaxespad=-1.5)
        
        self.cam_pitch_line, = self.canvas.axex2.plot(self.x, self.y, '.-', markersize=5, animated=True, label='cam_pitch', color='blue', lw=1)
        self.imu_pitch_line, = self.canvas.axex2.plot(self.x, self.y, '.-', markersize=5, animated=True, label='imu_pitch', color='red', lw=1)
        self.canvas.axex2.legend(loc='upper right', fontsize=7, ncol=3, shadow=True, borderaxespad=-1.5)

        self.cam_yaw_line, = self.canvas.axes3.plot(self.x, self.y, '.-', markersize=5,animated=True, label='cam_yaw', color='blue', lw=1)
        self.imu_yaw_line, = self.canvas.axes3.plot(self.x, self.y, '.-', markersize=5,animated=True, label='imu_yaw', color='red', lw=1)
        self.canvas.axes3.legend(loc='upper right', fontsize=7, ncol=3, shadow=True, borderaxespad=-1.5)

    def update_line2(self, i):
        cam_roll, cam_pitch, cam_yaw = cam_degrees[0], cam_degrees[1], -cam_degrees[2]
        imu_roll, imu_pitch, imu_yaw = imu_degrees[0], imu_degrees[1], imu_degrees[2]
        
        old_cam_roll = self.cam_roll_line.get_ydata()
        old_cam_pitch = self.cam_pitch_line.get_ydata()
        old_cam_yaw = self.cam_yaw_line.get_ydata()
        old_imu_roll = self.imu_roll_line.get_ydata()
        old_imu_pitch = self.imu_pitch_line.get_ydata()
        old_imu_yaw = self.imu_yaw_line.get_ydata()
            
        new_cam_roll = np.r_[old_cam_roll[1:], cam_roll]
        new_cam_pitch = np.r_[old_cam_pitch[1:], cam_pitch]
        new_cam_yaw = np.r_[old_cam_yaw[1:], cam_yaw]
        new_imu_roll = np.r_[old_imu_roll[1:], imu_roll]
        new_imu_pitch = np.r_[old_imu_pitch[1:], imu_pitch]
        new_imu_yaw = np.r_[old_imu_yaw[1:], imu_yaw]
    
        self.cam_roll_line.set_ydata(new_cam_roll)
        self.cam_pitch_line.set_ydata(new_cam_pitch)
        self.cam_yaw_line.set_ydata(new_cam_yaw)
        self.imu_roll_line.set_ydata(new_imu_roll)
        self.imu_pitch_line.set_ydata(new_imu_pitch)
        self.imu_yaw_line.set_ydata(new_imu_yaw)
        time.sleep(0.1)
        return [self.cam_roll_line, self.cam_pitch_line, self.cam_yaw_line, self.imu_roll_line, self.imu_pitch_line, self.imu_yaw_line]
        
    def on_start(self):
        self.ani = animation.FuncAnimation(self.canvas.figure, self.update_line2, blit=True, interval=25)

    def on_stop(self):
        self.ani._stop()
 
def imuCallBack(msg):
    global cam_callback_flag

    if cam_callback_flag:
        global imu_radians
        global imu_degrees
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)	
        
        ## radian to degree
        roll = math.degrees(round(euler[0],2))
        pitch = math.degrees(round(euler[1],2))
        yaw = math.degrees(round(euler[2],2))

        imu_radians = euler
        imu_degrees = roll, pitch*5, yaw
        cam_callback_flag = False

    
def camCallBack(msg):
    global cam_radians
    global cam_degrees
    global cam_callback_flag

    x = round(msg.data[0],2)
    y = round(msg.data[1],2)
    z = round(msg.data[2],2)

    roll = math.degrees(msg.data[0])
    pitch = math.degrees(msg.data[1])
    yaw = math.degrees(-msg.data[2])

    cam_radians = x,y,z
    cam_degrees = roll, pitch*5, yaw
    cam_callback_flag = True
    

def getROSmsg():
    rospy.Subscriber('/camera_Rt', Float64MultiArray , camCallBack, queue_size=1)
    rospy.Subscriber('/vectornav/IMU', Imu, imuCallBack, queue_size=1) 
     
if __name__ == "__main__":
    qApp = QApplication(sys.argv)
    rospy.init_node('qt_compare_imu_cam')
    cam_radians = []
    cam_degrees = []
    imu_radians = []
    imu_degrees = []
    cam_callback_flag = False

    while True:
        getROSmsg()
        # rospy.sleep(1)
        aw = AnimationWidget()
        aw.show()
        sys.exit(qApp.exec_())