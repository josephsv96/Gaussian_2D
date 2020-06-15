# https://hackaday.io/project/12384-autofan-automated-control-of-air-flow/log/41862-correcting-for-lens-distortions
# https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

import cv2
import numpy as np


def barrel_undist(img):

    height, width = img.shape[:2]

    # Define camera matrix (K)
    c_x = width/2.0     # define center x
    c_y = height/2.0    # define center x
    f_x = 1250.         # define focal length x
    f_y = 1250.         # define focal length x

    cam_mat = np.array([[f_x,     0.,     c_x],
                        [0.,      f_y,    c_y],
                        [0.,      0.,     1.]])

    # Define distortion coefficients (D) from calibrate.py
    # k1 => negative to remove barrel distortion
    k1 = -0.22168819
    k2 = -0.0805781
    p1 = 0.11702644
    p2 = -0.01788088
    k3 = 0.0341295

    dist_coeff = np.array([k1, k2, p1, p2, k3])

    # Generate new camera matrix from parameters
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
        cam_mat, dist_coeff, (width, height), 0)

    # Generate look-up tables for remapping the camera image
    mapx, mapy = cv2.initUndistortRectifyMap(
        cam_mat, dist_coeff, None, newcameramatrix, (width, height), 5)

    # Remap the original image to a new image
    newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    return newimg
