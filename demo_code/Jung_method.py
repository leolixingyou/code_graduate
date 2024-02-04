import cv2
import numpy as np

class MotionEstimation:
    def __init__(self):
        pass

    def get_motion_vector(self, img1, img2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        motion_vectors = [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in matches]
        return motion_vectors

    def get_intersection_point(self, motion_vectors):
        intersections = []
        for i in range(len(motion_vectors)):
            for j in range(i+1, len(motion_vectors)):
                line1 = motion_vectors[i]
                line2 = motion_vectors[j]
                intersection = np.cross(line1, line2)
                intersections.append(intersection)
        return intersections

    def get_lines(self, img):
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10, maxLineGap=10)
        return lines

    def get_axes(self, lines):
        if lines is None or len(lines) < 3:
            raise ValueError("Not enough lines detected to compute axes.")
        axes = lines[:3, 0, :]  
        return axes.reshape(3, 2, 2)  

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """计算从vec1到vec2的旋转矩阵"""
        # 扩展2D向量到3D
        vec1_3d = np.append(vec1, 0)
        vec2_3d = np.append(vec2, 0)
        
        a, b = (vec1_3d / np.linalg.norm(vec1_3d)).reshape(3), (vec2_3d / np.linalg.norm(vec2_3d)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix


    def get_rotation(self, axes, world_axes):
        directions_img = axes[:, 1, :] - axes[:, 0, :]
        directions_world = world_axes[:, 1, :] - world_axes[:, 0, :]
        R = self.rotation_matrix_from_vectors(directions_img[0], directions_world[0])
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        return np.rad2deg(np.array([x, y, z]))

    def main(self,img1,img2):
        estimator = MotionEstimation()
        motion_vectors = estimator.get_motion_vector(img1, img2)
        ###### for anaylze
        ### intersection_points = estimator.get_intersection_point(motion_vectors)

        lines = estimator.get_lines(img1)
        axes = estimator.get_axes(lines)

        world_axes = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]], [[0, 0], [0, -1]]])

        pitch, yaw, roll = estimator.get_rotation(axes, world_axes)
        return pitch, yaw, roll

if __name__ == "__main__":
    
    img1 = cv2.imread('pitch_0_yaw_0_0.png', 0)
    img2 = cv2.imread('pitch_9.8_yaw_0_1.png', 0)
    estimator = MotionEstimation()
    pitch, yaw, roll = estimator.main(img1, img2)
    print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

