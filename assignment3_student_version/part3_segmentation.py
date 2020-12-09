import os
import sys

import cv2 as cv
import numpy as np
# import kitti_dataHandler


def main(dist_thresh=5.0):
    ################
    # Options
    ################
    # Input dir and output dir
    depth_dir = 'data/train/gt_depth'
    label_dir = 'data/train/gt_labels'
    output_dir = 'data/train/est_segmentation'
    bb_dir = 'data/train/est_bb'
    est_depth_dir = 'data/train/est_depth'
    dirs = [output_dir]
    sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    dist_thresh = dist_thresh
    ################

    # Make dirs
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for sample_name in sample_list:
        print("#######################")
        print(f"Processing sample {sample_name}")
        # Read depth map
        # Depths less than 10 already discarded
        depth_map = cv.imread(f"{est_depth_dir}/{sample_name}.png", cv.IMREAD_GRAYSCALE)

        # Read 2d bbox
        boxes = np.load(f"{bb_dir}/{sample_name}.npy")
        print("Boxes: ")
        print(boxes)

        # For each bbox
        seg_mask = np.zeros_like(depth_map)
        seg_mask -= 1  # Set all pixels to 255 (we will set car pixels to 0 later on)
        for box in boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            # clip because some bb are out of image bounds
            x_min = np.clip(x, 0, depth_map.shape[1] - 1)
            x_max = np.clip(x+w, 0, depth_map.shape[1] - 1)
            y_min = np.clip(y, 0, depth_map.shape[0] - 1)
            y_max = np.clip(y+h, 0, depth_map.shape[0] - 1)
            # Estimate the average depth of the objects
            avg_depth = np.average(depth_map[y_min:y_max, x_min:x_max])
            # Find the pixels within a certain distance from the centroid
            for i in range(x_min, x_max + 1):
                for j in range(y_min, y_max + 1):
                    if abs(float(depth_map[j][i]) - avg_depth) < dist_thresh:
                        seg_mask[j][i] = 0

        # Save the segmentation mask
        cv.imwrite(f"{output_dir}/{sample_name}.png", seg_mask)


def test_results():
    est_seg_dir = 'data/train/est_segmentation'
    gt_seg_dir = 'data/train/gt_segmentation'
    sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']

    avg_precision = 0
    avg_recall = 0

    for sample_name in sample_list:
        print("########################")
        print(f"Processing sample {sample_name}")
        est_seg_map = cv.imread(f"{est_seg_dir}/{sample_name}.png", cv.IMREAD_GRAYSCALE)
        gt_seg_map = cv.imread(f"{gt_seg_dir}/{sample_name}.png", cv.IMREAD_GRAYSCALE)

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        # 0 = car, 255 = no car
        for i in range(gt_seg_map.shape[1]):
            for j in range(gt_seg_map.shape[0]):
                if gt_seg_map[j][i] < 255 and est_seg_map[j][i] < 255:
                    true_pos += 1
                elif gt_seg_map[j][i] == 255 and est_seg_map[j][i] == 255:
                    true_neg += 1
                elif gt_seg_map[j][i] == 255 and est_seg_map[j][i] < 255:
                    false_pos += 1
                elif gt_seg_map[j][i] < 255 and est_seg_map[j][i] == 255:
                    false_neg += 1

        precision = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)
        avg_precision += precision
        avg_recall += recall
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    avg_precision /= len(sample_list)
    avg_recall /= len(sample_list)
    return avg_precision, avg_recall

def search_best_precision():
    best_dist = 0
    best_precision = 0
    for dist_thresh in np.arange(1, 10, 0.5):
        main(dist_thresh)
        precision, recall = test_results()
        if precision > best_precision:
            best_dist = dist_thresh
            best_precision = precision
    print(f"Best_dist: {best_dist}, Best_precision: {best_precision}")

if __name__ == '__main__':
    main(5.5)
    test_results()

