import os
import sys

import cv2 as cv
import numpy as np
# import kitti_dataHandler


def get_segmentation_map(train=True, dist_thresh=5.0, crop_ratio=0.2):
    ################
    # Options
    ################
    # Set directories and samples
    if train:
        output_dir = 'data/train/est_segmentation'
        bb_dir = 'data/train/est_bb'
        est_depth_dir = 'data/train/est_depth'
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    else:
        output_dir = "data/test/est_segmentation"
        bb_dir = 'data/test/est_bb'
        est_depth_dir = 'data/test/est_depth'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    dirs = [output_dir]

    # Set options
    dist_thresh = dist_thresh
    crop_ratio = crop_ratio
    ################

    # Make dirs
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Iterate through all samples
    for sample_name in sample_list:
        print("#######################")
        print(f"Processing sample {sample_name}")
        # Read depth map
        # Depths less than 10 already discarded
        depth_map = cv.imread(f"{est_depth_dir}/{sample_name}.png", cv.IMREAD_GRAYSCALE)

        # Read 2d bbox
        boxes = np.load(f"{bb_dir}/{sample_name}.npy")
        print("Boxes: ")  # Print bounding boxes info
        print(boxes)

        # Create segmentation mask
        seg_mask = np.zeros_like(depth_map)
        seg_mask -= 1  # Set all pixels to 255 (we will set car pixels to 0 later on)
        # Iterate through all bounding boxes (already limited to cars only)
        for box in boxes:
            # Extract variables
            x, y, w, h = box[0], box[1], box[2], box[3]
            # clip because some bb are out of image bounds
            x_min = np.clip(x, 0, depth_map.shape[1] - 1)
            x_max = np.clip(x+w, 0, depth_map.shape[1] - 1)
            y_min = np.clip(y, 0, depth_map.shape[0] - 1)
            y_max = np.clip(y+h, 0, depth_map.shape[0] - 1)

            # tighten search range for avg value b/c pixels near edges are often not car
            x_min_search = int(x_min + (x_max - x_min) * crop_ratio)
            x_max_search = int(x_max - (x_max - x_min) * crop_ratio)
            y_min_search = int(y_min + (y_max - y_min) * crop_ratio)
            y_max_search = int(y_max - (y_max - y_min) * crop_ratio)

            # Estimate the average depth of the objects
            depth_map_search_crop = depth_map[y_min_search:y_max_search, x_min_search:x_max_search]
            avg_depth = np.sum(depth_map_search_crop)/np.count_nonzero(depth_map_search_crop)  # use count non-zero because zero depth values need to be discarded

            # Find the pixels within a certain distance from the avg depth
            for i in range(x_min, x_max + 1):
                for j in range(y_min, y_max + 1):
                    if abs(float(depth_map[j][i]) - avg_depth) < dist_thresh:
                        seg_mask[j][i] = 0

        # Save the segmentation mask
        cv.imwrite(f"{output_dir}/{sample_name}.png", seg_mask)

def test_results():
    # Set directories
    est_seg_dir = 'data/train/est_segmentation'
    gt_seg_dir = 'data/train/gt_segmentation'
    sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']

    # Track avg precision and recall
    avg_precision = 0
    avg_recall = 0

    # Iterate through all samples
    for sample_name in sample_list:
        print("########################")
        print(f"Processing sample {sample_name}")

        # Get segmentation maps
        est_seg_map = cv.imread(f"{est_seg_dir}/{sample_name}.png", cv.IMREAD_GRAYSCALE)
        gt_seg_map = cv.imread(f"{gt_seg_dir}/{sample_name}.png", cv.IMREAD_GRAYSCALE)

        # Counts for precision and recall calculations
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        # 0 = car, 255 = no car
        # Get tp/tn/fp/fn
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

        # Calculate precision and recall
        precision = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)

        # Update average precision and recall
        avg_precision += precision
        avg_recall += recall

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    # Calculate average precision and recall
    avg_precision /= len(sample_list)
    avg_recall /= len(sample_list)
    print(f"Avg Precision: {avg_precision} | Avg Recall: {avg_recall}")

    # Return average precision and recall
    return avg_precision, avg_recall

def search_best_dist_thresh(start, end, step):
    # Search through a range of distance thresholds for best one (based on precision)
    best_dist = 0
    best_precision = 0
    for dist_thresh in np.arange(start, end, step):
        get_segmentation_map(dist_thresh=dist_thresh)
        precision, recall = test_results()
        if precision > best_precision:
            best_dist = dist_thresh
            best_precision = precision
    print(f"Best_dist_thresh: {best_dist}, Best_precision: {best_precision}")

def search_best_crop_ratio(start, end, step):
    # Search through a range of crop ratios for best one (based on precision)
    best_crop_ratio = 0
    best_precision = 0
    for crop_ratio in np.arange(start, end, step):
        get_segmentation_map(crop_ratio=crop_ratio)
        precision, recall = test_results()
        if precision > best_precision:
            best_crop_ratio = crop_ratio
            best_precision = precision
    print(f"Best_crop_ratio: {best_crop_ratio}, Best_precision: {best_precision}")

if __name__ == '__main__':
    # Good distance thresholds 5, 5.5, 4.3
    # Good cropping ratios 0.2, 0.4
    get_segmentation_map(train=False, dist_thresh=4.3, crop_ratio=0.2)
    #avg_precision, avg_recall = test_results()  # Uncomment to test results of training set (requires results to be generated)
    #search_best_crop_ratio(0.0, 0.5, 0.05)  # Uncomment for crop ratio search
    #search_best_dist_thresh(2, 7, 0.1)  # Uncomment for dist threshold search

