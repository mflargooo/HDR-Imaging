from ccwb_tm import *
from cp_hw2 import writeHDR
from cv2 import imwrite

if __name__ == '__main__':
    '''
    ===========================================================================
    '''
    # the location and name of the image you are looking to color correct and tonemap
    hdr_path = "../data/church_stack"
    hdr_img_name = "jpg_gaussian_linear.hdr"

    # You can opt to or not to perform color correction or tonemapping here
    # NOTE: If tonemapping is selected, an untonemapped color correct image will still
    #       be saved. Tonemapped images are also saved after gamma encoding, while
    #       purely color corrected images are not!
    do_color_correction = False
    do_gray_world_wb = False
    do_rgb_tone_mapping = True
    do_luminance_tone_mapping = False

    patch_coords = np.array([(3352, 1453), (3361, 1290), (3339, 1150), (3339, 974), (3334, 816), (3334, 658), 
                            (3506, 1448), (3510, 1286), (3493, 1123), (3501, 983), (3484, 816), (3488, 658), 
                            (3664, 1448), (3659, 1290), (3655, 1136), (3646, 952), (3637, 812), (3637, 658), 
                            (3813, 1466), (3818, 1268), (3809, 1123), (3804, 974), (3804, 820), (3804, 671)])

    r, g, b = read_colorchecker_gm()
    true_colors = np.stack([r, g, b], axis=2).reshape((24, 3))

    # If tone mapping feel free to modfiy the values here!
    K = .005
    B = .5

    '''
    ===========================================================================
    '''

    img = readHDR(hdr_path + "/" + hdr_img_name)

    if do_color_correction:
        img = ccwb(img, true_colors, patch_coords, patch_size=5)
        writeHDR("%s/%s_%s.hdr" % (hdr_path, "color_checker_cc", hdr_img_name[:-4]), img)
        print("Finished color correction!")
    
    if do_gray_world_wb:
        img = gray_world(img)

        writeHDR("%s/%s_%s.hdr" % (hdr_path, "gray_world_cc", hdr_img_name[:-4]), img)
        print("Finished color correction!")

    if do_rgb_tone_mapping:
        tone_mapped = tonemap(img, K, B, mode='rgb')
        ge = gamma_encode(tone_mapped)
        cv2.imwrite("%s/tm_%s_%s.jpg" % (hdr_path, 'rgb', hdr_img_name[:-4]), np.uint8(255 * ge)[:,:,::-1])
        print("Finished %s tonemapping!" % 'rgb')

    if do_luminance_tone_mapping:
        tone_mapped = tonemap(img, K, B, mode='luminance')
        ge = gamma_encode(tone_mapped)
        cv2.imwrite("%s/tm_%s_%s.jpg" % (hdr_path, 'luminance', hdr_img_name[:-4]), np.uint8(255 * ge)[:,:,::-1])
        print("Finished %s tonemapping!" % 'luminance')

