import os
import sys

import cv2
import numpy as np
import kitti_dataHandler


def main():

    ################
    # Options
    ################
    # Input dir and output dir
    disp_dir = 'data/train/disparity'
    output_dir = 'data/train/est_depth'
    calib_dir = 'data/train/calib'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list):
        # Read disparity map

        # Read calibration info

        # Calculate depth (z = f*B/disp)

        # Discard pixels past 80m

        # Save depth map


if __name__ == '__main__':
    main()
