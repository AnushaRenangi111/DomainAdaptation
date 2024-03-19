import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import sys

# ----------------------------------------------------------------------------
def apply_equalization(img_x, type):
    if type == 'none':
        return img_x
    elif type == 'rgb':
        img_x = equalize_hist_rgb(img_x)
    elif type == 'hsv':
        img_x = equalize_hist_hsv(img_x)
    elif type == 'gray':
        img_x = equalize_hist_gray(img_x)
    elif type == 'gr_clahe':
        img_x = equalize_clahe_gray(img_x)
    elif type == 'co_clahe':
        img_x = equalize_clahe_lab(img_x)
    elif type == 'hsv_clahe':
        img_x = equalize_clahe_hsv(img_x)
    else:
        raise Exception('Undefined equalization type: ' + type)

    return img_x

# ----------------------------------------------------------------------------
def equalize_hist_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img

# ----------------------------------------------------------------------------
def equalize_hist_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    img = cv2.merge(eq_channels)
    return img

# ----------------------------------------------------------------------------
def equalize_hist_hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img)
    V = cv2.equalizeHist(V)
    img = cv2.merge([H, S, V])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

# ----------------------------------------------------------------------------
def equalize_clahe_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

# ----------------------------------------------------------------------------
def equalize_clahe_lab(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img

# ----------------------------------------------------------------------------
def equalize_clahe_hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    V = clahe.apply(V)
    img = cv2.merge([H, S, V])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img
