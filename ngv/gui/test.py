import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray

from queue import Queue
import inspect
import copy
import math
import os

queue_front_distance_gt = Queue()
queue_front_distance_before = Queue()
queue_front_distance_after = Queue()

g_imu_init_pose = None
g_imu_pose_flag = True
queue_degrees_imu = Queue()
g_degree_imu = [0,0,0]

g_front_distance_gt = -1.7
g_front_distance_before = -1.7
g_front_distance_after = -1.7
mean = -1.7 

qsize = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
filename='result'
file_ext='.csv'
toggle = True
output_path = ''

def front_distance_before_callback(msg):
    global toggle
    try:
        if queue_front_distance_before.qsize() < qsize:
            distance = copy.deepcopy(msg.data[0])
            queue_front_distance_before.put(distance)
            toggle = True

    except Exception as e:
        pass        
        #print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def front_distance_after_callback(msg):
    try:
        if queue_front_distance_after.qsize() < qsize:
            distance = copy.deepcopy(msg.data[0])
            queue_front_distance_after.put(distance)

    except Exception as e:
        pass        
        #print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def LiDARcallback(msg):
    try:
        if queue_front_distance_gt.qsize() < qsize:
            for marker in msg.markers:
                lidar_x = -marker.pose.position.x
                lidar_y = marker.pose.position.y
                lidar_z = marker.pose.position.z
            
            if len(msg.markers) > 0:
                GT_dist = math.sqrt(lidar_x**2+lidar_y**2)
                queue_front_distance_gt.put(GT_dist)

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def callback_imu(msg):
    global g_imu_pose_flag, g_imu_init_pose
    quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    cur_imu = [euler[0], euler[1], 0]

    if g_imu_pose_flag and cur_imu[1] != 0:
        g_imu_init_pose = cur_imu
        g_imu_pose_flag = False

    cur_imu = [euler[0]-g_imu_init_pose[0], euler[1]-g_imu_init_pose[1], 0]

    ## radian to degree
    roll = math.degrees(cur_imu[0])
    pitch = math.degrees(cur_imu[1])
    yaw = math.degrees(cur_imu[2])
    if queue_degrees_imu.qsize() < 2:
        queue_degrees_imu.put([roll, pitch, yaw])

def main():
    global g_front_distance_gt, g_front_distance_before, g_front_distance_after, f, toggle, g_degree_imu

    if queue_front_distance_gt.qsize() > qsize - 2:
        g_front_distance_gt = round(np.mean(np.array(queue_front_distance_gt.queue)),1)
        queue_front_distance_gt.get()
    else:
        pass

    if queue_front_distance_before.qsize() > qsize -2:
        g_front_distance_before = round(np.mean(np.array(queue_front_distance_before.queue)),1)
        queue_front_distance_before.get()
    else:
        pass

    if queue_front_distance_after.qsize() > qsize -2:
        g_front_distance_after = round(np.mean(np.array(queue_front_distance_after.queue)),1)
        queue_front_distance_after.get()
    else:
        pass

    if not queue_degrees_imu.empty():
        g_degree_imu = queue_degrees_imu.get()

    if toggle:
        print('\033[2;1H' + str(g_front_distance_gt) + '\n' + str(g_front_distance_before) + '\n' + str(g_front_distance_after))
        #f.write(str(g_front_distance_gt) + ',' + str(g_front_distance_before) + ',' + str(g_front_distance_after) +',\n')
        toggle = False


def logging_init():
    global output_path, filename, file_ext
    if not os.path.exists('{}'.format(dir_path) + '/results'):
       os.mkdirs('{}'.format(dir_path) + '/results')

    output_path='{}'.format(dir_path) + '/results/%s%s' %(filename,file_ext)
    uniq=1
    while os.path.exists(output_path):
        output_path = '{}'.format(dir_path) + '/results/%s_%d%s' % (filename,uniq,file_ext) 
        uniq+=1



if __name__=='__main__':
    rospy.init_node('GUI', anonymous=True)
    rospy.Subscriber('/lidar/postpoint', MarkerArray, LiDARcallback, queue_size=10)
    rospy.Subscriber('/detection_front/dist', Float32MultiArray, front_distance_after_callback, queue_size=10)
    rospy.Subscriber('/detection_front/ori_dist', Float32MultiArray, front_distance_before_callback, queue_size=10)
    logging_init()
    f = open(output_path, 'w')
    while not rospy.is_shutdown():
        main()
    f.close()
