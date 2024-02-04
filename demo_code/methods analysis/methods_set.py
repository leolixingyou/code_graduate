import os
import numpy as np
from sklearn.linear_model import LinearRegression

def linear_error(gt_p, errors):
    gt_p_array = np.array(gt_p)
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Fit the model to the data
    model.fit(gt_p_array, errors)
    
    # Return the trained model
    return model

def plane_f(gt_a,num,gt_p,error_list):
    p1,p2,p3,p4 = 0,0,0,0
    for id, con in enumerate(gt_a):
        if con == [num,num]:
            p1 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
        if con == [num,-num]:
            p2 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
        if con == [-num,-num]:
            p3 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
        if con == [-num,num]:
            p4 = np.array([gt_p[id][0],gt_p[id][1],error_list[id]])
    v1 = p3 - p1
    v2 = p2 - p1
    norm_p = np.cross(v1, v2)
    d = np.dot(norm_p, p1)
    if norm_p[2] != 0:
        plane_func = lambda x, y: (-norm_p[0]*x - norm_p[1]*y - d) / norm_p[2]
    else:
        v1 = p3 - p4
        v2 = p2 - p4
        norm_p = np.cross(v1, v2)
        d = np.dot(norm_p, p1)
        plane_func = lambda x, y: (-norm_p[0]*x - norm_p[1]*y - d) / norm_p[2]
    return plane_func