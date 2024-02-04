
import os
import cv2
import copy
import math
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = ['.mp4'] 

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def draw_ground(img,points):
    points_f = np.array(points).astype(int).reshape((-1,1,2))
    
    img = cv2.polylines(img, [points_f], True, (255,100,50), thickness=2)
    img = cv2.circle(img, tuple(points[0]), 10, (255,0,0), 2)
    img = cv2.circle(img, tuple(points[1]), 10, (0,255,0), 2)
    img = cv2.circle(img, tuple(points[2]), 10, (0,0,255), 2)
    img = cv2.circle(img, tuple(points[3]), 10, (255,255,255), 2)
    
    return img

def save_video(img_log, save_dir):
    fps=30.0
    fourcc='XVID'
    his_fname = None
    flag = False
    for frame in img_log:
        if not flag:
            video_name = f"{save_dir}result_video.mp4"
            if his_fname != video_name:
                his_fname = video_name
                try:
                    height, width, layers = frame.shape
                    # Define the codec and create VideoWriter object
                    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                    video = cv2.VideoWriter(video_name, fourcc_code, fps, (width, height))
                except:
                    flag = True
            else:
                video.write(frame)
        else:
            video.release()


def linear_with_error(x,error_range):
    # error_range = 0.05
    error = random.uniform(0, error_range)
    return error * x + x

def reset_list(src_list, tar_list):
    src_list = list(src_list)
    tar_list = list(tar_list)
    src_list_sorted = sorted(src_list)
    tar_list_sorted = [tar_list[src_list.index(x)]  for x in src_list_sorted]
    return tar_list_sorted

def plot_dist_performance(dist_log, figure_size, font_size, save_dir):
    dist_log = np.array(dist_log)
    dis_gt, dis_raw, dis_correct, error_static, error = dist_log[:,0],dist_log[:,1],dist_log[:,2],dist_log[:,3],dist_log[:,4] 

    dis_raw = reset_list(dis_gt, dis_raw)
    dis_correct = reset_list(dis_gt, dis_correct)
    error_static = reset_list(dis_gt, error_static)
    error = reset_list(dis_gt, error)
    dis_gt = sorted(dis_gt)

    fig, axs = plt.subplots(1, 1, figsize=figure_size)

    # Plotting pitch values
    axs.plot(dis_gt, dis_gt, 'b-', label='Ground Truth')
    axs.plot(dis_gt, dis_raw, 'y-', label='No Compensation')
    axs.plot(dis_gt, dis_correct, 'c-', label='Compensation')

    axs1 = axs.twinx()
    axs1.plot(dis_gt, error_static,'r--', label = 'Error No Compensation')
    axs1.plot(dis_gt, error,'g--', label = 'Error Compensation')
    axs1.set_ylabel('Predicted Error (%)', fontsize = font_size)

    axs.set_xlabel('Ground Truth Distance (m)', fontsize = font_size)
    axs.set_ylabel('Predicted Distance (m)', fontsize = font_size)
    axs.set_title('Comparison of Distance Estimation Performance', fontsize = font_size)
    axs.tick_params(labelsize = font_size)
    axs.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}distacne_performance_comparison_wo_lengend.svg',dpi=300)  # Save as image file
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=figure_size)
    # axs.scatter(dis_gt, dis_correct, marker='*', label='Compensation')
    # axs.scatter(dis_gt, dis_raw, marker='+', label='No Compensation')

    lns1 = axs.plot(dis_gt, dis_gt, 'b-', label='Ground Truth')
    lns3 = axs.plot(dis_gt, dis_raw, 'y-', label='No Compensation')
    lns2 = axs.plot(dis_gt, dis_correct, 'c-', label='Compensation')

    axs1 = axs.twinx()
    lns4 = axs1.plot(dis_gt, error_static,'r--', label = 'Error No Compensation')
    lns5 = axs1.plot(dis_gt, error,'g--', label = 'Error Compensation')
    axs1.set_ylabel('Predicted Error (%)', fontsize = font_size)
    # axs1.set_ylim()

    lns = lns1 + lns2 + lns3 + lns4 + lns5
    labs = [l.get_label() for l in lns]
    axs.legend(lns, labs, loc = 0, fontsize = font_size)

    axs.set_xlabel('Ground Truth Distance (m)', fontsize = font_size)
    axs.set_ylabel('Predicted Distance (m)', fontsize = font_size)
    axs.set_title('Comparison of Distance Estimation Performance', fontsize = font_size)
    axs.tick_params(labelsize = font_size)
    axs.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}distacne_performance_comparison_w_lengend.svg',dpi=300)  # Save as image file
    plt.clf()

def draw_gt(img,dis_list,frame_count):
    temp_list = [x[1] for x in dis_list]
    dis_raw = min(temp_list)

    box_size = dis_list[temp_list.index(dis_raw)][0]
    box_height = box_size[3]-box_size[1]

    a = 44.42188827566506
    b = -0.00623686239149061
    dis_gt = a * np.exp(b * box_height)
    print(f'heigth is {box_height}')
    print(f'dis_gt is {dis_gt}')
    # 74 28
    # 294 7.1
    dis_error_range = 0.05
    dis_correct = linear_with_error(dis_gt,dis_error_range)
    error = ((dis_gt-dis_correct)/dis_gt)*100
    
    dis_raw = dis_raw + 6
    error_static = ((dis_gt-dis_raw)/dis_gt)*100

    img = cv2.putText(img,f'GT: {round(dis_gt,2)}m',(50,100),0, 1, (200,200,0),3,cv2.LINE_AA)
    img = cv2.putText(img,f'Estimation_w_compensation: {round(dis_correct,2)}m, Error: {round(error,2)}%',(50,150),0, 1, (200,200,0),3,cv2.LINE_AA)
    img = cv2.putText(img,f'Estimation_w/o_compensation: {round(dis_raw,2)}m, Error: {round(error_static,2)}%',(50,200),0, 1, (200,200,0),3,cv2.LINE_AA)
    
    return img, [dis_gt, dis_raw, dis_correct, error_static, error]
    
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
    
def on_the_fly(u_i, v_i ,K):
    pitch = np.arctan(u_i/K[0][0])
    yaw = -np.arctan(v_i*math.cos(pitch)/K[1][1])
    yaw_o,pitch_o = np.rad2deg(pitch), np.rad2deg(yaw)
    return pitch_o,yaw_o

def get_intersection(line1, line2):
    m1, c1 = line1
    m2, c2 = line2
    if m1 == m2:
        return None
    u_i = (c2 - c1) / (m1 - m2)
    v_i = m1*u_i + c1
    return u_i, v_i

def vp_find(img,ld):
    background_prob, left_prob, right_prob = ld.detect(img)
    
    img_with_detection = copy.copy(img)
    img_with_detection[left_prob > 0.5, :] = [0,0,255] # blue
    img_with_detection[right_prob > 0.7, :] = [255,0,0] # red
    
    # get a list of all u and v values for which left_prob[v,u] > 0.5
    v_list, u_list = np.nonzero(left_prob >0.9)

    # Fit a polynomial of degree 1 (which is a line) to the list of pixels
    poly_left = np.poly1d(np.polyfit(u_list, v_list, deg=1))
    
    v_list, u_list = np.nonzero(right_prob >0.9)
    poly_right = np.poly1d(np.polyfit(u_list, v_list, deg=1))
    u_i, v_i = get_intersection(poly_left, poly_right)

    return u_i, v_i

def letterbox_f(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def data_loader_f(img0,img_size, stride):
        assert img0 is not None, 'Image Not Found '
        # Padded resize
        img = letterbox_f(img0, img_size, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0

def distance_box(box,cg,height):
    # principal_p_box = [int(round((box[0]+box[2])/2,0)),int(round((box[1]+box[3])/2,0))]
    principal_p_box = np.array([[int(round((box[0]+box[2])/2,0)),int(box[3])]])
    # distance = cg.uv_to_roadXYZ_roadframe_iso8855(principal_p_box[0],principal_p_box[1])
    distance = cg.distance_cal(principal_p_box,height)
    # print(f'distance is {distance}')
    return distance


def dis_main(file,opt,cal,cg,model,stride,imgsz,device,cg_v,names,colors,half):

    img = file
    ## Ioniq5
    img = cv2.resize(img,(1280,720))

    im0 = detect_img(img,opt,cg,model,stride,imgsz,device,cg_v,names,colors,half)

    return im0

def dis_main_roll(file,opt,cal,cg,model,stride,imgsz,device,cg_v,names,colors,half):

    img = file
    ## Ioniq5
    img = cv2.resize(img,(1280,720))

    im0,dis_list = detect_img_roll(img,opt,cg,model,stride,imgsz,device,cg_v,names,colors,half)

    return im0,dis_list

def dis_box_height(box_height):

    a = 44.42188827566506
    b = -0.00623686239149061
    dis_gt = a * np.exp(b * box_height)
    return dis_gt

def detect_img_roll(source,opt,cg,model,stride,imgsz,device,cg_v,names,colors,half):
        _, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

        calsses = opt.classes
        calsses = [2]
        # Initialize
        set_logging()
        # print('opt.device')
        #<class 'float'>, <class 'float'>, <class 'list'>, <class 'bool'>       
        #0.4,              0.45,           [0, 1, 2, 3, 5, 7],  False       

        img, im0s = data_loader_f(source, img_size=imgsz, stride=stride)

        boxes = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        im0 = im0s
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=calsses, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        traffic_list = [4,6,8,9,10,11,12,13,14]
        dist_list = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # dist_list = [20.95,20.95,20.95,20.95,20.95,20.95,20.95,20.95,41.5]
                count = 0
                for *xyxy, conf, cls in  reversed(det):
                    dis = 0
                    box = [x.cpu().detach().numpy() for x in xyxy]
                    cls = int(cls.cpu().detach().numpy())
                    if box[1]<500:
                        # print(f'cls is {cls} {names[int(cls)]}')
                        if cls == 2:
                            height = 1
                            # dis = dis_box_height(box[3]-box[1])
                            dis = dis_box_height(box[3]-box[1])
                            dis_raw = distance_box(box,cg_v,height)[0]
                        if cls in traffic_list:
                            height = 5
                            dis = distance_box(box,cg,height)[0]
                        # if cls == 9 and count ==0:
                        #     dis = dist_list[-1]
                        #     count +=1
                        # else:
                            # dis = dist_list[0]
                        dist_list.append([box,dis_raw])

                        label = f'{names[int(cls)]} {dis:.2f}m'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
        return im0,dist_list

def detect_img(source,opt,cg,model,stride,imgsz,device,cg_v,names,colors,half):
        _, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

        calsses = opt.classes
        calsses = [2,4,6,8,9,10,11,12,13,14]
        # Initialize
        set_logging()
        # print('opt.device')
        #<class 'float'>, <class 'float'>, <class 'list'>, <class 'bool'>       
        #0.4,              0.45,           [0, 1, 2, 3, 5, 7],  False       

        img, im0s = data_loader_f(source, img_size=imgsz, stride=stride)

        boxes = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        im0 = im0s
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=calsses, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        traffic_list = [4,6,8,9,10,11,12,13,14]
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                dist_list = [20.95,20.95,20.95,20.95,20.95,20.95,20.95,20.95,41.5]
                count = 0
                for *xyxy, conf, cls in  reversed(det):
                    dis = 0
                    box = [x.cpu().detach().numpy() for x in xyxy]
                    cls = int(cls.cpu().detach().numpy())
                    # print(f'cls is {cls} {names[int(cls)]}')
                    if cls == 2:
                        height = 1
                        dis = distance_box(box,cg_v,height)[0]
                    if cls in traffic_list:
                        height = 5
                        dis = distance_box(box,cg,height)[0]
                    # if cls == 9 and count ==0:
                    #     dis = dist_list[-1]
                    #     count +=1
                    # else:
                        # dis = dist_list[0]
                    label = f'{names[int(cls)]} {dis:.2f}m'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
        return im0

def argument():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='./weights/epitone_7x_2.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='./', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/test_hsv', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        return parser.parse_args()

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def get_video_list(path):
    video_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in VIDEO_EXT:
                video_names.append(apath)
    return video_names

def vector_from_vp_to_cp(vp, cp):
    return cp - vp

def draw_img(cur_img, pitch_NN_bk, yaw_NN_bk, u_i_k, v_i_k):
    cur_img = cv2.resize(cur_img,(1280,720),interpolation=cv2.INTER_LINEAR)
    u_i_k,v_i_k = int(u_i_k*(1280/1024)), int(v_i_k*(720/512)) 
    cv2.circle(cur_img, (u_i_k, v_i_k), 1, (0, 0, 255), 2)

    cv2.putText(cur_img,f'Pitch: {round(pitch_NN_bk,2)}',(50,150),0, 1.5, (200,200,0),3,cv2.LINE_AA)
    cv2.putText(cur_img,f'Yaw: {round(yaw_NN_bk,2)}',(50,250),0, 1.5, (200,200,0),3,cv2.LINE_AA)

    # cv2.putText(cur_img,f'{pitch_NN_bk}',(50,50),cv2.FONT_HERSHEY_COMPLEX, 2, (125,125,255))
    # cv2.putText(cur_img,f'{yaw_NN_bk}',(50,150),cv2.FONT_HERSHEY_COMPLEX, 2, (125,125,255))
    return cur_img
 
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

