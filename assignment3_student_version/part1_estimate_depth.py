import os
import sys

import cv2 as cv
import numpy as np
# import kitti_dataHandler

from assignment2_calibration_code import *

def main(train=True):

    ################
    # Options
    ################
    # Input dir and output dir
    if train:
        disp_dir = 'data/train/disparity'
        output_dir = 'data/train/est_depth'
        calib_dir = 'data/train/calib'
    else:
        disp_dir = 'data/test/disparity'
        output_dir = 'data/test/est_depth'
        calib_dir = 'data/test/calib'
    dirs = [disp_dir, output_dir, calib_dir]
    if train:
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    else:
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    # Make dirs
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Get depth map
    for sample_name in (sample_list):
        print(f"Processing depth for sample {sample_name}")
        # Read disparity map
        disp_map = cv.imread(f"{disp_dir}/{sample_name}.png", cv.IMREAD_GRAYSCALE)

        # Read calibration info
        frame_calib = read_frame_calib(calib_dir + '/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Calculate depth (z = f*B/disp)
        depth_map = np.empty_like(disp_map)
        for i in range(disp_map.shape[0]):
            for j in range(disp_map.shape[1]):
                if disp_map[i][j] == 0:
                    depth = 0
                else:
                    depth = stereo_calib.f * stereo_calib.baseline / disp_map[i][j]
                    if depth > 80 or depth < 0.1:  # Discard pixels past 80m
                        depth = 0
                depth_map[i][j] = depth

        # Save depth map
        cv.imwrite(f"{output_dir}/{sample_name}.png", depth_map)

def test_results():
    est_depth_dir = 'data/train/est_depth'
    gt_depth_dir = 'data/train/gt_depth'
    sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']

    for sample in sample_list:
        print("##################################")
        print(f"Processing sample {sample}")
        est_depth = cv.imread(f"{est_depth_dir}/{sample}.png", cv.IMREAD_GRAYSCALE)
        gt_depth = cv.imread(f"{gt_depth_dir}/{sample}.png", cv.IMREAD_GRAYSCALE)
        gt_depth[gt_depth > 80] = 0
        diff = cv.absdiff(est_depth, gt_depth)
        print(f"Average depth diff: {np.average(diff)}")
        print(f"Top10 depth diff: {np.sort(diff, None)[-10:]}")

if __name__ == '__main__':
    # main()
    test_results()
