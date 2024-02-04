import sys

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import uic
from std_msgs.msg import Float64MultiArray

import imugl
import ctrlros
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

form_class = uic.loadUiType("{}/ngv.ui".format(dir_path))[0]

class WindowClass(QMainWindow, form_class) :
	def __init__(self) :
		super(WindowClass, self).__init__()
		self.setupUi(self)
		self.imu_widget = imugl.ImuGL()
		self.imu_widget_2 = imugl.ImuGL2()

		self.imu_layout.addWidget(self.imu_widget)
		self.imu_layout_2.addWidget(self.imu_widget_2)

		self.ctrlros = ctrlros.CtrlRos()
		self.ctrlros.sendQE.connect(self.updateQE)
		self.ctrlros.sendRP.connect(self.updateRPY)

		self.ctrlros.sendRPfromCamera.connect(self.updateCAMRPY)

		self.pushButton.clicked.connect(self.initializebutton)

		self.imu_widget.show()
		self.imu_widget_2.show()
		

	@pyqtSlot(object)
	def updateQE(self, quaternion):
		quat_txt = "x : {}  y : {}  z : {}  w : {}".format(round(quaternion[0],1), round(quaternion[1],1), round(quaternion[2],1), round(quaternion[3],1))
		self.quat_label.setText(quat_txt)

	@pyqtSlot(float, float, float)
	def updateRPY(self, roll, pitch, yaw):
		rpy_txt = "roll : {} pitch : {} yaw : {} ".format(round(roll,1), round(pitch,1), round(yaw,1))
		self.rotation.setText(rpy_txt)

	@pyqtSlot(float, float, float)
	def updateCAMRPY(self, roll, pitch, yaw):
		cam_rpy_txt = "roll : {} pitch : {} yaw : {} ".format(round(roll,1), round(pitch,1), round(yaw,1))
		self.rotation_2.setText(cam_rpy_txt)

	def initializebutton(self):
		self.ctrlros.sendInitialize(True)

if __name__ == "__main__" :
	app = QApplication(sys.argv)
	myWindow = WindowClass()
	myWindow.show()
	app.exec_()
