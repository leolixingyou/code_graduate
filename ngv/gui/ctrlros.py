import rospy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, Imu
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform

class CtrlRos():
    def __init__(self):
        super(CtrlRos, self).__init__()
        rospy.init_node('GUI', anonymous=True)

        self.sub_topic()

    def sub_topic(self):
        rospy.init_node('GUI', anonymous=True)
        rospy.Subscriber('/cam0/compressed', CompressedImage, self.left_callback)   
        # rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.front_callback)   
        rospy.Subscriber('/od_result', Image, self.front_callback, queue_size=1)   
        rospy.Subscriber('/cam1/compressed', CompressedImage, self.right_callback)   
        
        rospy.Subscriber('/vectornav/IMU', Imu, self.imuCallBack, queue_size=10)
        rospy.Subscriber('/Camera/Transform', Transform, self.img_pose_callback, queue_size=10)
        
        rospy.Subscriber('/detection/dist', Float32MultiArray, self.front_distance_after_callback, queue_size=10)
        rospy.Subscriber('/detection/ori_dist', Float32MultiArray, self.front_distance_befor_callback, queue_size=10)
        rospy.Subscriber('/lidar/postpoint', MarkerArray, self.LiDARcallback, queue_size=10)
		

