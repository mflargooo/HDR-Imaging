import numpy as np
import cv2

def generate_ramp():
    size = 255
    scale = 5
    line = np.array([i // scale for i in range(size * scale)])
    return np.full((size * scale, size * scale), line, dtype=np.uint8)

def names_from_file(names_path):
    f = open(names_path, 'r')
    names = f.read().strip().split('\n')
    f.close()

    return names

'''
path is the directory where the images to use are and where the results will
be saved. names is an array that holds the image names to use, which are 
obtained from a line-separated file. described in the comment above '___main__'
'''

# MEAN IS RETURNED NOT UNNORMALIZED
def generate_mean(path, names):
    N = len(names)

    ref = cv2.imread("%s/%s" % (path, names[0]), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    s = 2 ** 16 - 1

    # compute average
    avg = np.zeros(ref.shape, dtype=np.float64)
    for i, name in enumerate(names):
        print("Creating average image %d/%d..." % (i + 1, N), end='\r')
        avg = avg + (cv2.imread("%s/%s" % (path, name), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float64) / s) / N
    return avg * s

# VARIANCE IS RETURNED NORMALIZED
def generate_variance(path, names, mean):
    N = len(names)

    ref = cv2.imread("%s/%s" % (path, names[0]), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    s = 2 ** 16 - 1
    mean_normed = mean.astype(np.float64) / s if mean.max() > 1 else mean

    # compute variance
    var = np.zeros(ref.shape, dtype=np.float64)
    for i, name in enumerate(names):
        print("Creating variance image %d/%d..." % (i + 1, N), end='\r')
        img = cv2.imread("%s/%s" % (path, name), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float64)
        diff = img / s - mean_normed
        var = var + (diff * diff) / (N - 1)
    
    return var