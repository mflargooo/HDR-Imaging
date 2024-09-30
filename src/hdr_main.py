from hdr import *
from cp_hw2 import writeHDR

if __name__ == '__main__':
    paths = {}
    weight_schemes = {}
    s = 2 ** 16 - 1

    '''
    ===========================================================================
    '''

    # Change your desired pixel cut-offs here!
    z_min, z_max = 0, 1

    # Add any other weighting schemes if you like. Your weighting function does not have
    # to strictly be a function of the pixel value z and the exposure time t_k, but when
    # adding it to the dictionary below, call your function via a lambda function with 
    # parameters (z, t_k) as below, where z is a channel and t_k is the exposure time
    weight_schemes['uniform'] = lambda z, t_k : weights_uniform(z, z_min, z_max)
    weight_schemes['tent'] = lambda z, t_k : weights_tent(z, z_min, z_max)
    weight_schemes['gaussian'] = lambda z, t_k : weights_gaussian(z, z_min, z_max)
    weight_schemes['photon'] = lambda z, t_k : weights_photon(z, z_min, z_max, t_k)

    # Dark frame path, is the location of the dark frame to use when mergin RAW 
    # images with noise-optimal weights! Make sure to note the shutter speed
    # you used to capture your dark frame stack
    data = np.load("../data/noise_calibration/f22_iso100_s4/per_channel_noise_calibration_params.npz")    
    df_file = "../data/noise_calibration/f22_iso100_s4/dark_frame/dark_frame.tiff"
    df_shutter_speed = 2 ** 2

    red_data, green_data, blue_data = data['red'], data['green'], data['blue']
    
    # BGR order!
    weight_schemes['noise-optimal'] = [(lambda z, t_k : weights_noise_opt(z, z_min, z_max, t_k, np.exp(blue_data[0]), np.exp(blue_data[1]))),
                                       (lambda z, t_k : weights_noise_opt(z, z_min, z_max, t_k, np.exp(green_data[0]), np.exp(green_data[1]))),
                                       (lambda z, t_k : weights_noise_opt(z, z_min, z_max, t_k, np.exp(red_data[0]), np.exp(red_data[1])))]
    
    '''
    YOU CAN GENERATE ALL COMBINATIONS AT ONCE, OR CHOOSE TO GENERATE ONE COMBINATION
    SEE COMMENTS TO CHANGE THE APPROPRIATE VARIABLES!
    '''

    # Choose whether you want to generate all HDR combinations. If you choose to 
    # generate a single combination, choose your weighting scheme via the
    # corresponding key in the weights_scheme dictionary
    all_combos = False
    # ---- UNUSED IF all_combos == True ----
    scheme = 'gaussian' # ensure this scheem is in the weights dictionary above!
    file_type = 'jpg'       # ensure this key is in the paths dictionary below!
    # --------------------------------------

    # PATHS

    # path where the N images, exposure{1,2,...,N}.<file_type>, exist in
    path_name = '../data/church_stack'
    
    # If you choose you choose to generate all combination, the output HDR image
    # names will describe the combination and the files will be generated in the
    # same older the data was pulled from.
    # If a single combination is chosen you are free to choose the destination
    output_path = path_name

    # if the exposure times increment by sequential powers of 2, you can
    # fill in the minimum and maximum exponents here
    small_exp_power = -5
    high_exp_power = 2

    # otherwise, directly list your times in the order the images will be processed
    times = [(2 ** (k - 1 + small_exp_power)) for k in range(1, high_exp_power-small_exp_power+2)]

    # Make sure that the files are named exposure*.<file_type> where * = 1,...,N 
    # and N is the number of images.
    #
    # If your exposure times are not sequential powers of 2, change the range of
    # the following to (1, N+1)
    paths['jpg'] = [('%s/exposure%d.jpg' % (path_name, k)) for k in range(1, high_exp_power-small_exp_power+2)]
    paths['tiff'] = [('%s/exposure%d.tiff' % (path_name, k)) for k in range(1, high_exp_power-small_exp_power+2)]

    '''
    ===========================================================================
    '''

    if all_combos:
        for p in paths.keys():
            if p != "tiff": continue
            for w in weight_schemes.keys():
                lin_hdr, log_hdr = None, None
                if w == 'noise-optimal':
                    if p == 'tiff':
                        print("Creating image with %s weights..." % w)
                        lin_hdr, log_hdr = create_hdr_noise_optimal(paths[p], times, df_file, df_shutter_speed, weight_schemes[w])
                    else:
                        continue
                else:
                    print("Creating image with %s weights..." % w)
                    reg_weights = (lambda z, t_k : 1) if w == 'photon' else None
                    lin_hdr, log_hdr = create_hdr(paths[p], times, weight_schemes[w], reg_weights, z_min, z_max)
                    
                writeHDR("%s/%s_%s_%s.hdr" % (path_name, p, w, 'linear'), lin_hdr[:,:,::-1])
                writeHDR("%s/%s_%s_%s.hdr" % (path_name, p, w, 'log'), log_hdr[:,:,::-1])
    else:
        lin_hdr, log_hdr = None, None
        if scheme == 'noise-optimal':
            if file_type != 'tiff': raise ValueError("Noise optimal weighting expects linear 16-bit tiff images!")
            print("Creating image with %s weights..." % scheme)
            lin_hdr, log_hdr = create_hdr_noise_optimal(paths[file_type], times, df_file, df_shutter_speed, weight_schemes[scheme])
        else:
            print("Creating image with %s weights..." % scheme)
            reg_weights = (lambda z, t_k : 1) if scheme == 'photon' else None
            lin_hdr, log_hdr = create_hdr(paths[file_type], times, weight_schemes[scheme], reg_weights, z_min, z_max)
        
        writeHDR("%s/%s_%s_%s.hdr" % (output_path, file_type, scheme, 'linear'), lin_hdr[:,:,::-1])
        writeHDR("%s/%s_%s_%s.hdr" % (output_path, file_type, scheme, 'log'), log_hdr[:,:,::-1])
