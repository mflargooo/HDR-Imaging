# Assignment 2
The different stages of this assignment exist in their own files, with the 
exception of color correction and tonemapping, whose functionality is
implemented in ccwb_tm.py

All the files *_main.py are the ones you will run to use the program. Those
without "_main" hold the functionalty for that task.

One of the first blocks in __main__ in all the *_main.py will see many 
different variables constituting directories, image paths, pixel cut-offs, 
etc. Each variable has some comment describing what it does and requires,
so it will be useful to read them. Once you populate the fields as you wish,
you may run that *_main.py file directly

==============================================================================

hdr_main.py.

This file sees the merging of an exposure stack. We describe the different
variables here
-- z_min, z_max : Are globally defined pixel cut-offs for weight functions and
                  such--globally in that if you want one common cut-off, you
                  can maintain it here, otherwise fill in the min and max as
                  you wish

-- weight_schemes : A dictionary holding different weighting schemes. The
                    independent weighting functions do not need to have
                    parameters exclusive to the pixel value z and the exposure
                    time t_k, but when entering them in the dictionary, the
                    function must be wrapped by a lambda including only z and 
                    t_k. We support lambdas of only this type

-- data, dark_frame_file : This is data corresponding to noise-optimal weights
                           which may only be used with tiffs.
                           data holds the calibration parameters created by
                           noise_calib_main.py and dark_frame_file holds the
                           dark frame image computed by the same script

-- all_combos : If all combos is true, the script will generate all 
                combinations of HDR images, otherwise, you must specify
                the scheme and file_type in the following two variables

-- path : The path that holds your exposure stack

-- output_path : The path where images are saved, by defaul = path

If you use sequential powers of 2 for your shutter speeds:

-- small_exp_power : is the exponent of your shortest shutter speed

-- high_exp_power : is the exponent of your longest shutter speed

-- times : a list that is auto populated for with the above powers. If you 
           do not use sequential powers of 2, you are required to fill it in 
           yourself

-- paths : dictionary of paths to each exposure image in the stack. These 
           images must correspond to the exposure times in the variable times

==============================================================================

ccwb_tm_main.py.

This file allows you to color correct and tonemap your images

-- hdr_path : is the path to the directory where the HDR image you want to 
              process is

-- hdr_img_name : the name of the HDR image in that above path

-- do_color_correction : specify whether you want to do color correction VIA
                         A COLOR CHECKER IN THE SCENE
                         If true, you must populate patch_coords with the 
                         center pixel location of the each square in color
                         checker, such that they align with the return value
                         of read_colorchecker_gm()

-- do_gray_world_wb : Whether or not to perform gray world white balancing

-- do_rgb_tone_mapping : Whether or not to produce an image from RGB tonemapping

-- do_luminance_tone_mapping : Whether or not to produce an image from luminance
                               tonemapping

-- K : the key value, controls the brightness in the final tonemap

-- B : the burn value, can be used to suppress contrast in the final tonemap

==============================================================================

noise_calib_main.py.

Handles color calibration as request of part 5. If a ramp is needed, you can 
generate it via the first # comment in the block. This script stores each step
as it goes along. When running it attempts to find previously saved data. If 
that data can not be found, it recomputes it, and then recomputes everything
after that, as it cannot guarantee it is using the same data to recompute.

-- overwrite : Regardless if the the data exists, recompute. You can also set
               this to true between different sections of the calibration
               process

This comment from noise_calib_main.py is for the following four variables

    IMPORTANT:

    Write the text file that holds the names of all the images in the given
    path to be used to compute the average. This can be done with 
        $ ls | grep -F .tiff > {output file} 
    when in the directory given by the path.

    Any variable *_path is the directory where the images to use are and 
    where the results will be saved. 

    Any variable *_names_file requests this file name

-- dark_path : the location the dark_names_file is in

-- dark_names_file : a file listing the names of the tiff images to be used
                     to obtain the dark_frame as stated above

-- ramp_path : the location the ramp_names_file is in

-- ramp_names_file : a file listing the names of the tiff images taken of
                     the ramp for the calibration process

-- noise_calib_path : the path to save all calibration results to

-- visualize_sub_pts : used to visualize the histogram of a few pixels in the
                       of the *_subtracted images generated by calibration

The follow are used for threading in the computation of the variance

-- num_workers : the number of workers to compute the variance

-- N : determines how many samples to take for linear-best-fit-line. A higher
       N sees a faster computation of the variance frame.

==============================================================================

Again, once you populate the corresponding variables, you are free to call
    the script directly