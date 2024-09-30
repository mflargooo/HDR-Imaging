import numpy as np  
import cv2
import matplotlib.pyplot as plt
from os.path import isfile

# THESE WEIGHTING FUNCTION OPERATE ON THE ENTIRE IMAGE
def weights_uniform(img, z_min, z_max):
    bounds = np.logical_and(z_min <= img, img <= z_max)
    return np.where(bounds, 1, 0).astype(np.float32)

def weights_tent(img, z_min, z_max):
    bounds = np.logical_and(z_min <= img, img <= z_max)
    return np.where(bounds, np.min([img, 1 - img], axis=0), 0)

def weights_gaussian(img, z_min, z_max):
    bounds = np.logical_and(z_min <= img, img <= z_max)
    return np.where(bounds, np.exp(-16 * (img - .5) ** 2), 0)

def weights_photon(img, z_min, z_max, t_k):
    bounds = np.logical_and(z_min <= img, img <= z_max)
    return np.where(bounds, t_k, 0).astype(np.float32)

# THIS IS A PER-CHANNEL WEIGHTING FUNCTION
def weights_noise_opt(channel, z_min, z_max, t_k, g, add_var):
    bounds = np.logical_and(z_min <= channel, channel <= z_max)
    return np.where(bounds, t_k * t_k / (g * channel + add_var), 0)

# from paper 3
def gsolve(paths, times, n, l, weights, reg_weights):
    P = len(paths)
    I = cv2.imread(paths[0])[::n,::n]
    N = I.shape[0] * I.shape[1]

    A = np.zeros((3*N*P+257, 256+3*N), dtype=np.float64)
    b = np.zeros((3*N*P+257,1), dtype=np.float64)

    Z = np.zeros(3*N*P, dtype=np.uint8)
    Z[:3*N] = np.ravel(I).reshape(3*N)

    for j in range(1,P):
        I = cv2.imread(paths[j])[::n,::n]
        Z[3*N*j:3*N*(j+1)] = np.ravel(I).reshape(3*N)
    
    k=0
    for i in range(3*N):
        for j in range(P):
            z = Z[j*3*N+i]
            wz = weights(float(z) / 255, times[j])
            bj = np.log(times[j])
            A[k,z] = wz
            A[k,256+i] = -wz
            b[k] = wz * bj
            k+=1

    A[k,128] = 1
    k+=1

    for i in range(254):
        rw = reg_weights(float(i+1) / 255, np.nan)
        A[k,i] = l * rw
        A[k,i+1] = -2 * l * rw
        A[k,i+2] = l * rw
        k+=1

    res = np.linalg.lstsq(A, b, rcond=None)[0]
    return res[:256], res[256:]

def linearize(paths, times, weights, reg_weights=None, z_min=.05, z_max=.95, title='Pixel Value versus Log Exposure', visualize=False):
    if reg_weights == None:
        reg_weights = weights
    g = gsolve(paths, times, 200, 200, weights, reg_weights)[0]
    g[:int(255 * z_min + 1)] = g[int(255 * z_min + 1)]
    g[int(255 * z_max):] = g[int(255 * z_max)]

    if visualize:
        x = np.arange(256)
        plt.title(title)
        plt.xlabel('pixel value')
        plt.ylabel('log exposure')
        plt.scatter(x, g[x])
        plt.show()
    return g

def merge_exposure_stack(paths, times, weights, g=[]):
    # weights should take two params, pixel value and exposure time in that order
    # if a g is given, then the paths are assumed to lead to jpg, if not then tiffs 
    P = len(paths)

    is_tiff = len(list(np.ravel(g))) == 0
    div = 2 ** 16 - 1 if is_tiff else 255

    # load first image beforehand to initialize arrays with correct shapes
    LDR = cv2.imread(paths[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    print("loaded path 0", end='\r')

    lin = np.float32(LDR) / div
    if not is_tiff:
        for c in range(lin.shape[2]):
            lin[:,:,c] = np.exp(g[LDR[:,:,c]]).reshape(lin.shape[:2])
    LDR = np.float32(LDR) / div
            
    w  = weights(LDR, times[0])

    lin_num = w * lin / times[0]
    log_num = w * (np.log(lin + 1e-7) - np.log(times[0]))
    denomin = w
    
    for k in range(1,P):
        LDR = cv2.imread(paths[k], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        print("loaded path %d" % k, end='\r')

        lin = np.float32(LDR) / div
        if not is_tiff:
            for c in range(lin.shape[2]):
                lin[:,:,c] = np.exp(g[LDR[:,:,c]]).reshape(lin.shape[:2])
        LDR = np.float32(LDR) / div

        w = weights(LDR, times[k])

        lin_num += w * lin / times[k]
        log_num += w * (np.log(lin + 1e-7) - np.log(times[k]))
        denomin += w

    
    lin_sol = lin_num / (denomin + 1e-7)
    log_sol = np.exp(log_num / (denomin + 1e-7))

    overex = np.logical_and(denomin <= 1e-7, LDR > .5)
    underex = np.logical_and(denomin <= 1e-7, LDR < .5)
    
    lin_max = lin_sol[np.logical_not(overex)].max()
    lin_min = lin_sol[np.logical_not(underex)].min()
    lin_sol = np.where(overex, lin_max, lin_sol)
    lin_sol = np.where(underex, lin_min, lin_sol)

    log_max = log_sol[np.logical_not(overex)].max()
    log_min = log_sol[np.logical_not(underex)].min()
    log_sol = np.where(overex, log_max, log_sol)
    log_sol = np.where(underex, log_min, log_sol)

    return lin_sol, log_sol

'''
Make sure that your channel weights are given in BGR order since we cv2 to load!
'''
def merge_exposure_stack_per_channel(paths, times, dark_frame_file, dark_frame_shutter_speed, channel_weights=[]):
    # weights should take two params, pixel value and exposure time in that order
    # if a g is given, then the paths are assumed to lead to jpg, if not then tiffs 
    P = len(paths)

    div = 2 ** 16 - 1

    dark_frame = cv2.imread(dark_frame_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED) 
    if dark_frame.max() > 1:
        dark_frame /= div
    dark_frame /= dark_frame_shutter_speed

    assert(np.all(np.logical_and(0 <= dark_frame, dark_frame <= 1)))


    # load first image beforehand to initialize arrays with correct shapes
    LDR = cv2.imread(paths[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32) / div
    print("loaded path 0", end='\r')

    LDR = LDR - dark_frame * times[0]
    lin = np.clip(np.copy(LDR), 0, 1)

    B = channel_weights[0](LDR[:,:,0], times[0])
    G = channel_weights[1](LDR[:,:,1], times[0])
    R = channel_weights[2](LDR[:,:,2], times[0])

    w = np.stack([B, G, R], axis=2)

    lin_num = w * lin / times[0]
    log_num = w * (np.log(lin + 1e-7) - np.log(times[0]))
    denomin = w
    
    for k in range(1,P):
        LDR = cv2.imread(paths[k], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32) / div
        print("loaded path %d" % k, end='\r')
        
        LDR = LDR - dark_frame * times[k]
        lin = np.clip(np.copy(LDR), 0, 1)

        B = channel_weights[0](LDR[:,:,0], times[k])
        G = channel_weights[1](LDR[:,:,1], times[k])
        R = channel_weights[2](LDR[:,:,2], times[k])

        w = np.stack([B, G, R], axis=2)

        lin_num += w * lin / times[k]
        log_num += w * (np.log(lin + 1e-7) - np.log(times[k]))
        denomin += w

    lin_sol = lin_num / (denomin + 1e-7)
    log_sol = np.exp(log_num / (denomin + 1e-7))
    print(denomin.min(), denomin.max())

    overex = np.logical_and(denomin <= 1e-7, LDR > .5)
    underex = np.logical_and(denomin <= 1e-7, LDR < .5)
    
    lin_max = lin_sol[np.logical_not(overex)].max()
    lin_min = lin_sol[np.logical_not(underex)].min()
    lin_sol = np.where(overex, lin_max, lin_sol)
    lin_sol = np.where(underex, lin_min, lin_sol)

    log_max = log_sol[np.logical_not(overex)].max()
    log_min = log_sol[np.logical_not(underex)].min()
    log_sol = np.where(overex, log_max, log_sol)
    log_sol = np.where(underex, log_min, log_sol)

    return lin_sol, log_sol

# more straightforward function to create hdr images
def create_hdr(paths, times, weights, reg_weights=None, z_min=.05, z_max=.95):
    first = paths[0]
    i = len(first) - 1
    file_type = ""
    while (i >= 0):
        if first[i] == '.':
            break
        file_type = first[i] + file_type
        i -= 1

    print("Processing %d files of type %s..." % (len(paths), file_type))
    
    g = []
    if file_type == 'jpg':
        g = linearize(paths, times, weights, reg_weights, z_min, z_max)
    res = merge_exposure_stack(paths, times, weights, g)

    print("Finished!                            ")
    return res

'''
Make sure that your channel weights are given in BGR order since we cv2 to load!
'''
def create_hdr_noise_optimal(paths, times, dark_frame_file, dark_frame_shutter_speed, channel_weights=[]):
    if (not isfile(dark_frame_file)):
        raise ValueError("Could not find dark frame at location " + dark_frame_file)
    if (len(channel_weights) != 3):
        raise ValueError("Expected a list of three weighting functions for each channel!")
    
    first = paths[0]
    i = len(first) - 1
    file_type = ""
    while (i >= 0):
        if first[i] == '.':
            break
        file_type = first[i] + file_type
        i -= 1

    if file_type != "tiff": raise ValueError("Expected images of type tiff for noise optimal weights!")

    print("Processing %d files of type %s..." % (len(paths), file_type))
    
    res = merge_exposure_stack_per_channel(paths, times, dark_frame_file, dark_frame_shutter_speed, channel_weights)
    
    print("Finished!                            ")
    return res

'''
=======================================================================================
'''