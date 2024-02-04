import rosbag
from sensor_msgs.msg import Imu, CompressedImage
from gps_common.msg import GPSFix
from novatel_oem7_msgs.msg import INSPVA
import numpy as np
import cv2
bag = rosbag.Bag('bags/seg-2023-03-04-18-19-04.bag')

topics = ['/gmsl_camera/dev/video0/compressed', '/gps/gps', '/gps/imu', '/novatel/oem7/inspva']


timestamps = [1677921546.313612223,
            1677921554.462213755,
            1677921649.369158268,
            1677921663.048843861,
            1677921683.995615959,
            1677921688.969473839]

distances = [{topic: 999 for topic in topics} for _ in range(len(timestamps))]
min_data = [{topic: None for topic in topics} for _ in range(len(timestamps))]

## Select Nearest Topics
for topic, msg, t in bag.read_messages(topics=topics):
    t = t.to_sec()
    for i, ts in enumerate(timestamps):
        diff = np.abs(t - ts)
        if diff < distances[i][topic]:
            distances[i][topic] = diff
            min_data[i][topic] = msg

## Parse

directory = 'scenes'
def save_image(filename, msg):
    # Convert compressed image message to ROS image message
    filename = f"{directory}/{filename}.png"

    np_arr = np.frombuffer(msg.data, np.uint8)
    # Convert ROS image message to numpy array
    img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Save the image as a PNG file
    cv2.imwrite(filename, img_np)


def save_text(filename, msg):
    with open(f"{directory}/{filename}.yaml", 'w') as f:
        f.write('---\n')
        f.write(str(msg))

for i, ts in enumerate(timestamps):
    data = min_data[i]
    for topic in topics:
        filename = f"scene-{i}-{topic[1:].replace('/', '_')}"
        msg = data[topic]
        
        if topic == '/gmsl_camera/dev/video0/compressed':
            save_image(filename, msg)
            print('save image')
        else:
            print('save text')
            save_text(filename, msg)

bag.close()
