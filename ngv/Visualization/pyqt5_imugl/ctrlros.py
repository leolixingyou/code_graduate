from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QObject

import rospy
import rospkg
import roslib
import tf
import math

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image,CompressedImage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
import cv2
import numpy as np

from scipy.spatial.transform import Rotation 

class CtrlRos(QObject):
	sendRP = pyqtSignal(float, float, float)
	sendQE = pyqtSignal(object)
	sendRPfromCamera = pyqtSignal(float, float, float)

	def __init__(self) :
		super(CtrlRos, self).__init__()
		rospy.init_node('visualizer', anonymous=True)
		self.sub_topic()

	def sub_topic(self) :
		rospy.Subscriber('/vectornav/IMU', Imu, self.imuCallBack, queue_size=10)
		rospy.Subscriber('/camera_Rt', Float64MultiArray , self.camCallBack, queue_size=10)

	def imuCallBack(self, msgs) :
		"""
		#calculate
		quaternion = (data.orientation.x,data.orientation.y, data.orientation.z,data.orientation.w  )
		euler = tf.transformations.euler_from_quaternion(quaternion)	
		PI = 3.141592
		pitch = round(euler[0]*180/PI, 4)
		roll = round(euler[1]*180/PI, 4)
		yaw = round(euler[2]*180/PI, 4)
		"""
		
		## quaternion to euler(pitch, roll, yaw)
		quaternion = (msgs.orientation.x, msgs.orientation.y, msgs.orientation.z, msgs.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)	

		#Q_t = Rotation.from_quat(quaternion)	
	
		
		## radian to degree
		roll = math.degrees(euler[0])
		pitch = math.degrees(euler[1])
		yaw = math.degrees(euler[2])
		# print(roll, pitch, yaw)
		# print(euler)
		# print("-"*50)
		
		self.sendRP.emit(roll, pitch*5, yaw)
		self.sendQE.emit(quaternion)
		
	def camCallBack(self, msgs) :
		# print(x,y, z)
		# print("#"*50)
		
		self.cam_roll = math.degrees(msgs.data[0])
		self.cam_pitch = math.degrees(msgs.data[1])
		self.cam_yaw = math.degrees(msgs.data[2])
		# print(msgs.data[0], msgs.data[1], msgs.data[2])
		# print(self.cam_roll, self.cam_pitch, self.cam_yaw)
		# print("#"*50)
		self.sendRPfromCamera.emit(self.cam_roll, self.cam_pitch*5, self.cam_yaw)
		# self.sendRPfromCamera.emit(roll, pitch, yaw)

	def sendInitialize(self, flag) :
		



