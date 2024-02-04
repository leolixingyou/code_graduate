from PyQt5.QtGui import *
from PyQt5.QtWidgets import  *
from PyQt5.QtCore import *
from PyQt5 import uic

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, Imu
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import math
import tf

import sys
import os
#import inspect
from queue import Queue


dir_path = os.path.dirname(os.path.realpath(__file__))
form_class = uic.loadUiType("{}/mainwindow.ui".format(dir_path))[0]


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super(WindowClass, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("PyNGV")

        self.queue_qimg_front = Queue()
        self.queue_qimg_left = Queue()
        self.queue_qimg_right = Queue()

        img_bg = QImage("{}/resources/bg.png".format(dir_path))
        self.setWindowIcon(QIcon("{}/resources/icon.png".format(dir_path)))
        background = img_bg.scaled(QSize(1920, 1080))
        palette = QPalette()
        palette.setBrush(10, QBrush(background))
        self.setPalette(palette)


        self.left_distance_gt = 1
        self.left_distance_before = 1
        self.left_distance_after = 1
        self.front_distance_gt = 1
        self.front_distance_before = 1
        self.front_distance_after = 1
        self.right_distance_gt = 1
        self.right_distance_before = 1
        self.right_distance_after = 1

        self.plt_figure = plt.Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.plt_figure)
        self.figure_x_range = np.arange(50)
        self.figure_y_range = np.arange(-10,10,0.4)
        self.initUI()


        self.imu_init_pose = None
        self.imu_pose_flag = True
        self.queue_degrees_imu = Queue()
        self.degree_imu = [0,0,0]
        self.queue_degrees_img = Queue()
        self.degree_img = [0,0,0]

        ########### 아래에는 코드를 추가하지 말것 ############

        self.timer_drawing = QTimer(self)
        self.timer_drawing.start(10)
        self.timer_drawing.timeout.connect(self.mytimer)

        rospy.init_node('GUI', anonymous=True)
        rospy.Subscriber('/cam0/compressed', CompressedImage, self.callback_cam_left)
        rospy.Subscriber('/od_result', Image, self.callback_cam_front) 
        rospy.Subscriber('/cam1/compressed', CompressedImage, self.callback_cam_right)
        rospy.Subscriber('/vectornav/IMU', Imu, self.callback_imu, queue_size=10)
        rospy.Subscriber('/Camera/Transform', Transform, self.callback_img_pose, queue_size=10)
        
        #rospy.Subscriber('/detection/dist', Float32MultiArray, self.front_distance_after_callback, queue_size=10)
        #rospy.Subscriber('/detection/ori_dist', Float32MultiArray, self.front_distance_befor_callback, queue_size=10)
        #rospy.Subscriber('/lidar/postpoint', MarkerArray, self.LiDARcallback, queue_size=10)


    def initUI(self):
        self.pose_plot2.addWidget(self.figure_canvas)
        self.plt_figure.set_facecolor('#1C343E')
        self.plt_figure.tight_layout()         
        # 1~1 중 1번째(1,1,1)  서브 챠트 생성
        self.ax = self.plt_figure.add_subplot(111)      
        self.ax.set_facecolor('#628A9C')        
        # self.ax.set(xlabel='ff', ylabel='kk')
        self.ax.tick_params(axis='x', color='#628A9C', labelcolor='#628A9C')
        self.ax.get_xaxis().set_visible(False)     
        # 2D line
        self.imu_line, = self.ax.plot(self.figure_x_range, self.figure_y_range, label='IMU', color='#1C343E')
        self.cam_line, = self.ax.plot(self.figure_x_range, self.figure_y_range, label='CAM', color='#2BD2F8')     
        self.ax.legend(loc='upper left', frameon=False, ncol=2, facecolor='#628A9C')

        # 애니메이션 챠트 생성
        self.ani = animation.FuncAnimation(self.plt_figure, self.callback_animate, init_func=self.initPlot, interval=100, blit=False, save_count=50)
        self.figure_canvas.draw()

    def initPlot(self):
        self.imu_line.set_ydata([np.nan]*len(self.figure_x_range))
        self.cam_line.set_ydata([np.nan]*len(self.figure_x_range))
        return [self.imu_line, self.cam_line]   
 
    def callback_animate(self, i):
        if not self.queue_degrees_imu.empty():
            self.degree_imu = self.queue_degrees_imu.get()

        if not self.queue_degrees_img.empty():
            self.degree_img = self.queue_degrees_img.get()

        imu_pitch = self.degree_imu[1]
        cam_pitch = self.degree_img[1]

        old_imu_pitch = self.imu_line.get_ydata()
        old_cam_pitch = self.cam_line.get_ydata()            

        new_imu_pitch = np.r_[old_imu_pitch[1:], imu_pitch]
        new_cam_pitch = np.r_[old_cam_pitch[1:], cam_pitch]
        
        self.imu_line.set_ydata(new_imu_pitch)
        self.cam_line.set_ydata(new_cam_pitch)

        return [self.imu_line, self.cam_line]

    def callback_imu(self, msg):
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        cur_imu = [euler[0], euler[1], 0]

        if self.imu_pose_flag and cur_imu[1] != 0:
            self.imu_init_pose = cur_imu
            self.imu_pose_flag = False

        cur_imu = [euler[0]-self.imu_init_pose[0], euler[1]-self.imu_init_pose[1], 0]

        ## radian to degree
        roll = math.degrees(round(cur_imu[0],2))
        pitch = math.degrees(round(cur_imu[1],2))
        yaw = math.degrees(round(cur_imu[2],2))
        # print(pitch)
        self.queue_degrees_imu.put([roll, pitch, yaw])

    def callback_img_pose(self, msg):
        roll = -math.degrees(round(msg.rotation.x,2))
        pitch = -math.degrees(round(msg.rotation.y,2))
        yaw = -math.degrees(round(msg.rotation.z,2))
        # self.img_degrees = roll, pitch, yaw
        self.queue_degrees_img.put([roll, pitch, yaw])


    def mytimer(self):
        if not self.queue_qimg_front.empty():
            qimg = self.queue_qimg_front.get()
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(self.ui_front_img.size())
            self.ui_front_img.setPixmap(pixmap)
            self.set_text(self.front_distance_gt, self.front_distance_before, self.front_distance_after,'front')

        if not self.queue_qimg_left.empty():
            qimg = self.queue_qimg_left.get()
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(self.ui_left_img.size())
            self.ui_left_img.setPixmap(pixmap)
            self.set_text(self.left_distance_gt, self.left_distance_before, self.left_distance_after,'left')

        if not self.queue_qimg_right.empty():
            qimg = self.queue_qimg_right.get()
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(self.ui_right_img.size())
            self.ui_right_img.setPixmap(pixmap)
            self.set_text(self.right_distance_gt, self.right_distance_before, self.right_distance_after,'right')

    @pyqtSlot()
    def callback_cam_front(self, msg):
        img = QImage(msg.data, msg.width, msg.height, QImage.Format_RGB888)
        if self.queue_qimg_front.empty():
            self.queue_qimg_front.put(img)

    @pyqtSlot()
    def callback_cam_left(self, msg):
        img = QImage.fromData(msg.data)
        if self.queue_qimg_left.empty():
            self.queue_qimg_left.put(img)

    @pyqtSlot()
    def callback_cam_right(self, msg):
        img = QImage.fromData(msg.data)
        if self.queue_qimg_right.empty():
            self.queue_qimg_right.put(img)

    def text_color(self, label, value):
        if value <= 5 and value >= -5:
            label.setStyleSheet("Color : rgb(43, 210, 248)")
        else:
            label.setStyleSheet("Color : rgb(249, 85, 155)")

    def cal_error(self, gt, distance):
        error = 100*(abs(distance)-abs(gt))/abs(gt)
        return round(error, 1)

    def set_text(self, distance_gt, distance_before, distance_after, mode):
        error_before = self.cal_error(distance_gt, distance_before)
        error_after = self.cal_error(distance_gt, distance_after)
        if mode == 'left':
            self.ui_left_gt.setText(str(distance_gt))
            self.ui_left_before.setText(str(distance_before))
            if error_before < 1000:
                self.ui_left_error_before.setText(str(error_before))
            else:
                self.ui_left_error_before.setText('over')
            self.ui_left_after.setText(str(distance_after))
            if error_after < 1000:
                self.ui_left_error_after.setText(str(error_after))
            else:
                self.ui_left_error_after.setText('over') 
            self.text_color(self.ui_left_error_after, error_after)
        elif mode =='front':
            self.ui_front_gt.setText(str(distance_gt))
            self.ui_front_before.setText(str(distance_before))
            if error_before < 1000:
                self.ui_front_error_before.setText(str(error_before))
            else:
                self.ui_front_error_before.setText('over')
            self.ui_front_after.setText(str(distance_after))
            if error_after < 1000:
                self.ui_front_error_after.setText(str(error_after))
            else:
                self.ui_front_error_after.setText('over')
            self.text_color(self.ui_front_error_after, error_after)
        elif mode == 'right':
            self.ui_right_gt.setText(str(distance_gt))
            self.ui_right_before.setText(str(distance_before))
            if error_before < 1000:
                self.ui_right_error_before.setText(str(error_before))
            else:
                self.ui_right_error_before.setText('over')          
            self.ui_right_after.setText(str(distance_after))
            if error_after < 1000:
                self.ui_right_error_after.setText(str(error_after))
            else:
                self.ui_right_error_after.setText('over')
            self.text_color(self.ui_right_error_after, error_after)
        else:
            pass


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.showFullScreen()
    app.exec_()
