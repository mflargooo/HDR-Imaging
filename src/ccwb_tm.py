import numpy as np
import cv2
import matplotlib.pyplot as plt
from cp_hw2 import *

def ccwb(img, expected_colors, patch_centers, patch_size=3):
    pxs = len(patch_centers)
    A = np.ones((pxs, 4), dtype=np.float32)
    b = np.ones((pxs, 4), dtype=np.float32)

    pad = patch_size // 2
    for i in range(pxs):
        x, y = patch_centers[i]
        A[i,:3] = np.mean(img[y-pad:y+pad,x-pad:x+pad], axis=(0,1))
        b[i,:3] = expected_colors[i]
    
    #white patch
    # red, green, blue = A[18,:3]
    # max_c = max(red, green, blue)

    M = np.linalg.lstsq(A, b, rcond=None)[0]
    M = M[:3,:]

    homo = np.ones((img.shape[0], img.shape[1], 4))
    homo[:,:,:3] = img
    vec = np.reshape(homo, (img.shape[0]*img.shape[1], 4))
    out_vec = M @ vec.T
    cc = np.reshape(out_vec.T, (img.shape[0], img.shape[1], 3))
    # not a fan
    # cc[:,:,0] *= max_c / red
    # cc[:,:,1] *= max_c / green
    # cc[:,:,2] *= max_c / blue

    return cc

def gray_world(img):
    r_avg, g_avg, b_avg = np.average(img, axis=(0,1))

    scale = np.ones_like(img)
    scale[:,:,0] = g_avg / b_avg
    scale[:,:,1] = 1
    scale[:,::2] = g_avg / r_avg
    
    return img * scale

def tonemap(hdr, K=.15, B=.95, mode="rgb"):
    N = hdr.shape[0] * hdr.shape[1]

    if mode == "rgb":
        I_m = np.exp(np.sum(np.log(hdr + 1e-7) / N))
        I_hdr = K / I_m * hdr
        I_white = B * np.max(I_hdr)

        return I_hdr * (1 + I_hdr / (I_white * I_white)) / (1 + I_hdr)
    
    elif mode == "luminance":
        XYZ = lRGB2XYZ(hdr)

        denom = np.sum(XYZ, axis=2)
        x, y, Y = XYZ[:,:,0] / denom, XYZ[:,:,1] / denom, XYZ[:,:,1]

        I_m = np.exp(np.sum(np.log(Y + 1e-7) / N))
        I_hdr = K / I_m * Y
        I_white = B * np.max(I_hdr)

        Y = I_hdr * (1 + I_hdr / (I_white * I_white)) / (1 + I_hdr)
        X = (x * Y) / y
        Z = (1 - x - y) * Y / y

        return XYZ2lRGB(np.stack([X,Y,Z], axis=2))

    return ValueError("Expected mode \"rgb\" or \"luminance\"")

def gamma_encode(img):
    ge = np.zeros_like(img)
    less = img <= .0031308
    ge[less] = 12.92 * img[less]
    ge[np.logical_not(less)] = 1.055 * (img[np.logical_not(less)] ** (1 / 2.4)) - .055
    
    return np.clip(ge, 0, 1)