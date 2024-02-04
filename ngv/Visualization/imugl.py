"""
pip install PyOpenGL PyOpenGL_accelerate
https://github.com/pyside/Examples/blob/master/examples/opengl/samplebuffers.py
"""
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys, math

import myobjloader as myloader

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QOpenGLWidget
import ctrlros

class ImuGL(QOpenGLWidget) :
	def  __init__(self, parent = None):
		super(ImuGL, self).__init__(parent)
		self.ctrlros = ctrlros.CtrlRos()
		self.ctrlros.sendRP.connect(self.updateRP)
		self.roll = 0.0
		self.pitch = 0.0
		self.yaw = 0.0

	def initializeGL(self):
		glutInit()
		glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
		glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
		glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
		glEnable(GL_LIGHT0)
		glEnable(GL_LIGHTING)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_COLOR_MATERIAL)
		glShadeModel(GL_SMOOTH)
		glClearColor(1.0,1.0,1.0,1.0)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(45, 320/float(210), 1, 100.0)
		glEnable(GL_DEPTH_TEST)
		glMatrixMode(GL_MODELVIEW)

		self.car = myloader.ObjLoader("Car.obj")
		

	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glLoadIdentity()
		glPushMatrix()
		glScalef(0.3, 0.4, 0.4)
		glTranslatef(0.0,-0.3,-5.5)
		glRotatef(30,1,0,0)
		glRotatef(-45,0,1,0)

		self.drawLine()

		glRotatef(self.roll, 0, 1, 0)
		glRotatef(self.pitch*10+5 , -1, 0, 0)
		glCallList(self.car.gl_list)
		glPopMatrix()
	
	@pyqtSlot(float, float, float)
	def updateRP(self, roll_change, pitch_change, yaw_change):
		self.roll = roll_change
		self.pitch = pitch_change
		self.yaw = yaw_change
		self.update()

	def drawLine(self) :
		xroll = "R_{}".format(round(self.roll,1))
		ypitch = "P_{}".format(round(self.pitch,1))
		zyaw = "Y_{}".format(round(self.yaw,1))

		glPushMatrix()

		glPushMatrix();
		glColor3f(0.823, 0.929, 0.313);
		glBegin(GL_LINES);
		glVertex3f(0.0, 0.0, 5.0);
		glVertex3f(0.0, 0.0, -5.0);
		glEnd();
		self.drawBitmapText(xroll, 0.0, 0.5, 3.0);
		glPopMatrix();
		
		glPushMatrix();
		glColor3f(0.313, 0.737, 0.929);
		glBegin(GL_LINES);
		glVertex3f(5.0, 0.0, 0.0);
		glVertex3f(-5.0, 0.0, 0.0);
		glEnd();
		self.drawBitmapText(ypitch, -5.0, 0.0, 0.0);
		glPopMatrix();

		glPushMatrix();
		glColor3f(0.313, 0.929, 0.615);
		glBegin(GL_LINES);
		glVertex3f(0.0, 5.0, 0.0);
		glVertex3f(0.0, -5.0, 0.0);
		glEnd();
		self.drawBitmapText(zyaw, 0.0, 2.0,-0.2);
		glPopMatrix();

		glPopMatrix();

		glFlush();

	def drawBitmapText(self, str, x, y, z) :
		glRasterPos3f(x, y, z)
		for ch in str :
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(ch)))

	def resizeGL(self, width, height):
		glGetError()
		aspect = width if (height == 0) else width / height
		glViewport(0, 0, width,height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(50, aspect, 1, 100.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

class ImuGL2(QOpenGLWidget) :
	def  __init__(self, parent = None):
		super(ImuGL2, self).__init__(parent)
		self.ctrlros = ctrlros.CtrlRos()
		self.ctrlros.sendRPfromCamera.connect(self.updateRP)
		self.roll = 0.0
		self.pitch = 0.0
		self.yaw = 0.0

	def initializeGL(self):
		glutInit()
		glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
		glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
		glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
		glEnable(GL_LIGHT0)
		glEnable(GL_LIGHTING)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_COLOR_MATERIAL)
		glShadeModel(GL_SMOOTH)
		glClearColor(1.0,1.0,1.0,1.0)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(45, 320/float(210), 1, 100.0)
		glEnable(GL_DEPTH_TEST)
		glMatrixMode(GL_MODELVIEW)

		self.car = myloader.ObjLoader("Car.obj")
		

	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glLoadIdentity()
		glPushMatrix()
		glScalef(0.3, 0.4, 0.4)
		glTranslatef(0.3,-0.3,-5.5)
		glRotatef(30,1,0,0)
		glRotatef(-45,0,1,0)

		self.drawLine()

		glRotatef(self.roll, 0, 1, 0)
		glRotatef(self.pitch*10+5, -1, 0, 0)
		glCallList(self.car.gl_list)
		glPopMatrix()
	
	@pyqtSlot(float, float, float)
	def updateRP(self, roll_change, pitch_change, yaw_change):
		self.roll = roll_change
		self.pitch = pitch_change
		self.yaw = yaw_change
		self.update()

	def drawLine(self) :
		xroll = "R_{}".format(round(self.roll,1))
		ypitch = "P_{}".format(round(self.pitch,1))
		zyaw = "Y_{}".format(round(self.yaw,1))

		glPushMatrix()

		glPushMatrix();
		glColor3f(0.823, 0.929, 0.313);
		glBegin(GL_LINES);
		glVertex3f(0.0, 0.0, 5.0);
		glVertex3f(0.0, 0.0, -5.0);
		glEnd();
		self.drawBitmapText(xroll, 0.0, 0.5, 3.0);
		glPopMatrix();
		
		glPushMatrix();
		glColor3f(0.313, 0.737, 0.929);
		glBegin(GL_LINES);
		glVertex3f(5.0, 0.0, 0.0);
		glVertex3f(-5.0, 0.0, 0.0);
		glEnd();
		self.drawBitmapText(ypitch, -5.0, 0.0, 0.0);
		glPopMatrix();

		glPushMatrix();
		glColor3f(0.313, 0.929, 0.615);
		glBegin(GL_LINES);
		glVertex3f(0.0, 5.0, 0.0);
		glVertex3f(0.0, -5.0, 0.0);
		glEnd();
		self.drawBitmapText(zyaw, 0.0, 2.0,-0.2);
		glPopMatrix();

		glPopMatrix();

		glFlush();

	def drawBitmapText(self, str, x, y, z) :
		glRasterPos3f(x, y, z)
		for ch in str :
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(ch)))

	def resizeGL(self, width, height):
		glGetError()
		aspect = width if (height == 0) else width / height
		glViewport(0, 0, width,height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(50, aspect, 1, 100.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()


 
	