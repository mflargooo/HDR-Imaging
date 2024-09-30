from noise_calib import *
import matplotlib.pyplot as plt
from os.path import isfile
import threading

if __name__ == '__main__':
    '''
    ===========================================================================
    '''

    # cv2.imwrite("../data/ramp.png", generate_ramp())

    # This code saves each step as it runs; if it finds an existing file
    # use it instead of re-computing. If overwrite is True, then we always
    # perform the computation and ignore pre-computed values
    overwrite = False
    

    '''
    IMPORTANT:

    Write the text file that holds the names of all the images in the given
    path to be used to compute the average. This can be done with 
        $ ls | grep -F .tiff > {output file} 
    when in the directory given by the path.

    Any variable *_path is the directory where the images to use are and 
    where the results will be saved. 

    Any variable *_names_file requests this file name
    '''

    dark_path = "../data/noise_calibration/f22_iso100_s4/dark_frame"
    dark_names_file = "dark_names.txt"

    ramp_path = "../data/noise_calibration/f22_iso100_s4/ramp"
    ramp_names_file = "ramp_names.txt"

    # all intermediate and final calibration results are searched for and 
    # stored here
    noise_calib_path = "../data/noise_calibration/f22_iso100_s4"

    # visualize some subtracted points?
    visualize_sub_pts = False

    # used for threading the computation of the variance
    num_workers = 16

    # higher N = less samples to take in calibration, larger the N
    # the faster the workers
    N = 10
    
    '''
    ===========================================================================
    '''

    s = 2 ** 16 - 1

    # generate dark frame if it doesnt already exist or if write over is true
    dark_frame = None
    if (not overwrite and isfile(dark_path + "/dark_frame.tiff")):
        dark_frame = cv2.imread(dark_path + "/dark_frame.tiff", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float64)
    else:
        # if we are creating a new dark frame, we cannot guarantee that the
        # we are creating the same dark frame, so we have to write over everything 
        overwrite = True
        dark_names = names_from_file("%s/%s" % (dark_path, dark_names_file))
        dark_frame = generate_mean(dark_path, dark_names).astype(np.float64)
        cv2.imwrite(dark_path + "/dark_frame.tiff", dark_frame)
    print("Finished creating dark frame!                  ")

    # processing ramp images 
    # create subtracted images if they don't already exist or if write over is true
    ramp_names = names_from_file("%s/%s" % (ramp_path, ramp_names_file))
    sub_names = []
    N = len(ramp_names)

    for i, name in enumerate(ramp_names):
        sub_names.append("%s%s.tiff" % (name[:-5], "_subtracted"))
        sub_path = "%s/%s" % (ramp_path, sub_names[-1]) 
        if (not overwrite and isfile(sub_path)): 
            print("Skipped subtract %s, %d/%d" % (name, i + 1, N), end='\r')
            continue
        ramp_img = cv2.imread("%s/%s" % (ramp_path, name), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float64)
        cv2.imwrite(sub_path, ramp_img - dark_frame)
        print("Created %s %d/%d!" % (sub_names[-1], i+1, N), end='\r')
    print("Finished image subtraction!                    ")

    # get average and variance
    avg = None
    if (not overwrite and isfile(noise_calib_path + "/subtracted_average.npy")):
        avg = np.load(noise_calib_path + "/subtracted_average.npy")
    else:
        overwrite = True
        avg = generate_mean(ramp_path, sub_names)
        np.save(noise_calib_path + "/subtracted_average.npy", avg.astype(np.float64))
    print("Finished creating average frame!               ")
    
    var = None
    if (not overwrite and isfile(noise_calib_path + "/subtracted_variance.npy")):
        var = np.load(noise_calib_path + "/subtracted_variance.npy")
    else:
        overwrite = True
        var = generate_variance(ramp_path, sub_names, avg)
        np.save(noise_calib_path + "/subtracted_variance.npy", var)
    print("Finished creating variance frame!              ")
    
    # get rounded mean and average variance
    blue_data, green_data, red_data = None, None, None
    if (not overwrite and isfile(noise_calib_path + "/rounded_mean-average_variance.npz")):
        data = np.load(noise_calib_path + "/rounded_mean-average_variance.npz")
        blue_data, green_data, red_data = data['blue'], data['green'], data['red']
    else:
        print("Computing rounded mean and average variance for each channel...")
        A = np.round(avg[::N,::N,:])
        V = var[::N,::N,:]

        results = []
        def worker(id, c, r_avg, avg_var):
            l = len(r_avg)
            for i in range(id, l, num_workers):  
                avg_var[i] = np.mean(V[:,:,c], where=A[:,:,c] == r_avg[i])
                print("Worker %d processed element %d/%d" % (id, i, l))

        for c in range(A.shape[2]):
            r_avg = np.unique(A[:,:,c])
            avg_var = [0] * len(r_avg)

            threads = []
            for id in range(num_workers):
                thread = threading.Thread(target=worker, args=(id, c, r_avg, avg_var))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            
            print("Done processing channel " + str(c))

            results.append([np.array(r_avg) / s, np.array(avg_var)])
        np.savez(noise_calib_path + "/rounded_mean-average_variance.npz", red=results[2], green=results[1], blue=results[0])
        blue_data, green_data, red_data = results
    print("Finished creating rounded mean and average variance!                                         ")
    
    # ---- Histogram of 4 different pixels ----
    if visualize_sub_pts:
        fig = plt.figure()
        ax = fig.subplots(2, 2)
        x, y = dark_frame.shape[0] // 3, dark_frame.shape[1] // 3
        pixels = [(0, 0, 0), (x, y, 1), (2 * x, 2 * y, 2), (3 * x, 3 * y, 0)]
        bins = [np.zeros((2 ** 16 - 1), dtype=int) for _ in pixels]
        for i, pix in enumerate(pixels):
            ax[i // 2, i % 2].set_title("Histogram of values of pixel " + str(pix) + " across the stack")
        
        for name in sub_names:
            img = cv2.imread(ramp_path + "/" + name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            for i, pix in enumerate(pixels):
                bins[i][int(img[pix])] += 1

        fig.suptitle("Histogram of subtracted pixels")
        for i, pix in enumerate(pixels):
            ax[i // 2, i % 2].scatter(np.arange(2 ** 16 - 1), bins[i])
        plt.show()
    # ------------------------------------------

    results = []
    for channel, [rounded_avg, average_var] in [("Blue", blue_data), ("Green", green_data), ("Red", red_data)]:
        # cut off outliers
        r_avg_mean = np.mean(rounded_avg)
        r_avg_stddev = np.sqrt(np.var(rounded_avg))

        cutoff = np.logical_and(r_avg_mean - 4 * r_avg_stddev <= rounded_avg, rounded_avg <= 1.2 * r_avg_stddev + r_avg_mean)
        r_avg = np.log(rounded_avg[cutoff])
        avg_var = np.log(average_var[cutoff])

        # given the relationship between mean and variance, we perform lstsq, where
        # the slope give us the gain g and the intercept gives us the additive variance
        # This linear relationship sees the means as inputs and variances as outputs
        A = np.stack([r_avg, np.ones_like(r_avg)], axis=1)
        b = avg_var
        # g and var_add are returned unnormalized, we normalize before saving
        g, var_add = np.linalg.lstsq(A, b, rcond=None)[0]
        results.append([g, var_add])
        print("Normalized %s:\n ISO gain:          %f\n Additive variance: %f" % (channel, g, var_add))

        plt.xlabel("Rounded Mean")
        plt.ylabel("Average Variance")
        plt.title(channel)
        plt.scatter(r_avg, avg_var, c='black')
        plt.axline((0, var_add), (1, (var_add + g)), c=channel.lower())
        plt.show()

    np.savez(noise_calib_path + "/per_channel_noise_calibration_params.npz", red=results[2], green=results[1], blue=results[0])