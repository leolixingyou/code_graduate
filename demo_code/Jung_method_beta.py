import cv2
import numpy as np

class MotionEstimation_beta:
    def __init__(self):
        pass

    def get_motion_vector(self, img1, img2):
        fast = cv2.FastFeatureDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        kp1 = fast.detect(img1, None)
        kp1, des1 = brief.compute(img1, kp1)
        
        kp2 = fast.detect(img2, None)
        kp2, des2 = brief.compute(img2, kp2)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        motion_vectors = [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in good]
        return motion_vectors

    def get_intersection_point(self, motion_vectors):
        # 这里需要一个更复杂的算法来计算交点，但为了简化，我们只返回前两个motion vector的交点
        return motion_vectors[0][0]

    def get_lines(self, img):
        lsd = cv2.createLineSegmentDetector(0)
        print(lsd)
        lines, _, _, _ = lsd.detect(img)
        return lines

    def three_line_ransac(self, lines, core_intersection):
        # 这里需要一个3-Line RANSAC算法的实现，但为了简化，我们只返回前三条线
        # 可以使用 core_intersection 来帮助确定或验证 RANSAC 算法的输出
        return lines[:3]

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """计算从vec1到vec2的旋转矩阵"""
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
        # directions_img = axes[:, 1, :] - axes[:, 0, :]
        directions_img = axes[:, 0, 2:] - axes[:, 0, :2]

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

    def main(self, img1, img2):
        motion_vectors = self.get_motion_vector(img1, img2)
        core_intersection = self.get_intersection_point(motion_vectors)
        lines = self.get_lines(img1)
        best_axes = self.three_line_ransac(lines, core_intersection)
        world_axes = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]], [[0, 0], [0, -1]]])
        pitch, yaw, roll = self.get_rotation(best_axes, world_axes)
        return pitch, yaw, roll

if __name__ == "__main__":
    img1 = cv2.imread('pitch_0_yaw_0_0.png', 0)
    img2 = cv2.imread('pitch_9.8_yaw_0_1.png', 0)
    estimator = MotionEstimation()
    pitch, yaw, roll = estimator.main(img1, img2)
    print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
