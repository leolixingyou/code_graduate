
import os
import random
import numpy as np

def linear_with_error(x):
    error_range = 0.05
    error = random.uniform(0, error_range)
    return error * x + x

def get_intrinsic_matrix(field_of_view_deg, image_width, image_height):
    # For our Carla camera alpha_u = alpha_v = alpha
    # alpha can be computed given the cameras field of view via
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu],
                     [0, alpha, Cv],
                     [0, 0, 1.0]])

def get_py_from_vp(u_i, v_i, K):
    p_infinity = np.array([u_i, v_i, 1])
    K_inv = np.linalg.inv(K)
    r3 = K_inv @ p_infinity    
    r3 /= np.linalg.norm(r3)
    yaw = np.arctan2(r3[0], r3[2])
    pitch = -np.arcsin(r3[1])
    return pitch, yaw

def get_vp_from_py(pitch, yaw, K):
    r3=[0.,0.,0.]
    r3[1] = np.sin(np.deg2rad(-pitch))
    r3[0] = np.tan(np.deg2rad(yaw))
    r3[2] = 1.0

    vps = K @ r3
    u_gt = vps[0] / vps[2]
    v_gt = vps[1] / vps[2]
    return u_gt, v_gt

def get_intersection(line1, line2):
    m1, c1 = line1
    m2, c2 = line2
    if m1 == m2:
        return None
    u_i = (c2 - c1) / (m1 - m2)
    v_i = m1*u_i + c1
    return u_i, v_i

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

def vector_from_vp_to_cp(vp, cp):
    return cp - vp

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
