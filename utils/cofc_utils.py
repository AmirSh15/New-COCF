import numpy as np
import pickle as pkl
import os, time, itertools, argparse
import cv2
from matplotlib import pyplot as plt
from utils.face_feats import get_deep_feature, get_deep_features, initialize_deep_models

class face_element:
    def __init__(self, fno, bbox, img, feat):
        self.fno = fno
        self.bbox = bbox
        self.img = img
        self.feat = feat

def overlap_in_percent(ref, new):
    lft = max(new[0]-new[2]/2, ref[0]-ref[2]/2)
    rt = min(new[0]+new[2]/2, ref[0]+ref[2]/2)
    top = max(new[1]-new[3]/2, ref[1]-ref[3]/2)
    bot = min(new[1]+new[3]/2, ref[1]+ref[3]/2)
    if rt-lft < 0 or bot-top < 0:
        return 0.0
    else:
        return 100.0 * (rt-lft)*(bot-top) / (0.0001 + max(new[2]*new[3], ref[2]*ref[3]))

def euc_dist_sq(x1,x2):
    return np.sum((x1-x2)**2)
        
        
def display_cv_image(im, size=(10,10), label=""):
    if(len(im.shape) == 3):
        plt.figure(figsize=size)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.title(label)
        plt.show()
    else:
        plt.figure(figsize=size)
        plt.imshow(im, cmap="gray")
        plt.title(label)
        plt.show()


def get_face_bboxes_in_frame(im, model):
    rets = model.detector.detect_face(im)
    if rets is None:
        N = 0
        return np.zeros((N, 4), int)
    else:
        N = len(rets[0])
    np_rects = np.zeros((N, 4), int)
    for i, d in enumerate(rets[0]):
        np_rects[i,:] = np.array([d[0],d[1],d[2],d[3]])
    return np_rects

def shot_boundary(ppf, pf, f, thresh=60):
    #this method taken from Johan Mathe's code: https://github.com/johmathe/shotdetect
    d1 = np.mean(np.abs(ppf -pf))
    d2 = np.mean(np.abs(pf - f))
    diff = abs(d1-d2)
    
    if diff>thresh and d2>thresh:
        return True
    else:
        return False

def extract_bboxes_and_features(ls_frames, model):
    shot_data = []
    ii=0
    for f in ls_frames:
        fno = f[0]        
        frame = f[1]
        boxes = get_face_bboxes_in_frame(frame, model)
        if(boxes.shape[0] == 0):
            continue 
        for i in range(boxes.shape[0]):
            y1 = max(0,boxes[i,1])
            x1 = max(0, boxes[i,0])
            y2 = min(frame.shape[0], boxes[i,3]) + 1
            x2 = min(frame.shape[1], boxes[i,2]) + 1
            
            im = frame[y1:y2, x1:x2].copy()
            feat = get_deep_feature(im, model)
            shot_data.append(face_element(fno, boxes[i], im, feat))
        ii+=1
    return shot_data 
