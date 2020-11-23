import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os

# Shichen Lu Added
import glob
import matplotlib

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib

def q1():
    ## Input
    img_paths = glob.glob("./test/left/*") + glob.glob("./training/left/*")

    ## Output
    result_imgs_dir = "./result_imgs/"
    if not os.path.exists(result_imgs_dir):
        os.makedirs(result_imgs_dir)

    ## Detect Keypoints
    for img_path in img_paths:
        # Already grayscale
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        # Feature detector
        sift_detector = cv.SIFT_create(nfeatures=1000)
        kp, des = sift_detector.detectAndCompute(img, None)
        img_with_kp = cv.drawKeypoints(img, kp, None,
                                            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(result_imgs_dir + f'{img_path[2:].replace("/", "_")}_keypoints.jpg', img_with_kp)


def q2(training = True):
    ## Input
    if training:
        left_image_dir = os.path.abspath('./training/left')
        right_image_dir = os.path.abspath('./training/right')
        calib_dir = os.path.abspath('./training/calib')
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    else:
        left_image_dir = os.path.abspath('./test/left')
        right_image_dir = os.path.abspath('./test/right')
        calib_dir = os.path.abspath('./test/calib')
        sample_list = ['000011', '000012', '000013', '000014', '000015']


    ## Find keypoints + matches
    for sample_name in sample_list:
        left_image_path = left_image_dir + '/' + sample_name + '.png'
        right_image_path = right_image_dir + '/' + sample_name + '.png'

        # Already grayscale
        img_left = cv.imread(left_image_path, cv.IMREAD_GRAYSCALE)
        img_right = cv.imread(right_image_path, cv.IMREAD_GRAYSCALE)

        # Get features
        sift_detector = cv.SIFT_create(nfeatures=1000)
        kp_left, des_left = sift_detector.detectAndCompute(img_left, None)
        kp_right, des_right = sift_detector.detectAndCompute(img_right, None)
        # Fix bug where sometimes the detector returns 1001 features
        kp_left, des_left, kp_right, des_right = kp_left[0:1000], des_left[0:1000], kp_right[0:1000], des_right[0:1000]

        # Matching
        bf_matcher = cv.BFMatcher(normType=cv.NORM_L1)
        # Create mask to enforce epipolar constraint and positive disparities only
        mask_mtx = np.zeros((1000, 1000), dtype=np.uint8)
        for i in range(1000):
            for j in range(1000):
                if kp_left[i].pt[0] < kp_right[j].pt[0]: continue
                if abs(kp_left[i].pt[1] - kp_right[j].pt[1]) < 1:
                    mask_mtx[i][j] = 1
        matches = bf_matcher.match(des_left, des_right, mask_mtx)

        # Read calibration
        frame_calib = read_frame_calib(calib_dir + '/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        for i, match in enumerate(matches):
            pt_left = kp_left[match.queryIdx].pt
            pt_right = kp_right[match.trainIdx].pt
            disparity = pt_left[0] - pt_right[0]
            pixel_u_list.append(round(pt_left[0]))
            pixel_v_list.append(round(pt_left[1]))
            disparity_list.append(disparity)
            depth_list.append(stereo_calib.f * stereo_calib.baseline / disparity)

        ## Output
        result_matches_dir = "./result_matches/"
        if not os.path.exists(result_matches_dir):
            os.makedirs(result_matches_dir)

        with open(f"{result_matches_dir}P2_results_{'training' if training else 'test'}_{sample_name}.txt", "a") as output_file:
            output_file.truncate(0)
            for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
                line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
                output_file.write(line + '\n')

        # Draw matches
        result_imgs_dir = "./result_imgs/"
        if not os.path.exists(result_imgs_dir):
            os.makedirs(result_imgs_dir)

        img_with_matches = cv.drawMatches(img_left, kp_left, img_right, kp_right, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(f"{result_imgs_dir}P2_{'training' if training else 'test'}_{sample_name}_matches.png", img_with_matches)
        plt.imshow(img_with_matches)
        plt.show()

def q3(training = True):
    ## Input
    if training:
        left_image_dir = os.path.abspath('./training/left')
        right_image_dir = os.path.abspath('./training/right')
        calib_dir = os.path.abspath('./training/calib')
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    else:
        left_image_dir = os.path.abspath('./test/left')
        right_image_dir = os.path.abspath('./test/right')
        calib_dir = os.path.abspath('./test/calib')
        sample_list = ['000011', '000012', '000013', '000014', '000015']


    ## Find keypoints + matches
    for sample_name in sample_list:
        left_image_path = left_image_dir + '/' + sample_name + '.png'
        right_image_path = right_image_dir + '/' + sample_name + '.png'

        # Already grayscale
        img_left = cv.imread(left_image_path, cv.IMREAD_GRAYSCALE)
        img_right = cv.imread(right_image_path, cv.IMREAD_GRAYSCALE)

        # Get features
        sift_detector = cv.SIFT_create(nfeatures=1000)
        kp_left, des_left = sift_detector.detectAndCompute(img_left, None)
        kp_right, des_right = sift_detector.detectAndCompute(img_right, None)
        # Fix bug where sometimes the detector returns 1001 features
        kp_left, des_left, kp_right, des_right = kp_left[0:1000], des_left[0:1000], kp_right[0:1000], des_right[0:1000]

        # Matching
        bf_matcher = cv.BFMatcher(normType=cv.NORM_L1)
        mask_mtx = np.zeros((1000, 1000), dtype=np.uint8)
        for i in range(1000):
            for j in range(1000):
                if kp_left[i].pt[0] < kp_right[j].pt[0]: continue
                else: mask_mtx[j][i] = 1
        matches = bf_matcher.knnMatch(des_right, des_left, k=2, mask=mask_mtx)

        # Outlier Rejection
        pts_left = []
        pts_right = []
        good_matches = []
        for i in range(len(matches)):
            if len(matches[i]) == 0:  # Check for empty matches
                continue
            elif len(matches[i]) == 1:  # If only one match, add to good_matches
                match1 = matches[i][0]
                good_matches.append(match1)
                pts_left.append(kp_left[match1.trainIdx].pt)
                pts_right.append(kp_right[match1.queryIdx].pt)
            elif len(matches[i]) == 2:  # If multiple matches, use Lowe's ratio test
                match1, match2 = matches[i]
                if match1.distance < match2.distance*0.5:
                    good_matches.append(match1)
                    pts_left.append(kp_left[match1.trainIdx].pt)
                    pts_right.append(kp_right[match1.queryIdx].pt)
        pts_left = np.array(pts_left)
        pts_right = np.array(pts_right)
        F, mask = cv.findFundamentalMat(pts_left, pts_right, cv.FM_RANSAC, 1, 0.99, 10000)
        matches = np.array(good_matches)[mask.ravel() == 1]
        print(f"Good matches: {len(good_matches)}")
        print(f"Matches after RANSAC: {len(matches)}")

        # Read calibration
        frame_calib = read_frame_calib(calib_dir + '/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        for i, match in enumerate(matches):
            pt_left = kp_left[match.trainIdx].pt
            pt_right = kp_right[match.queryIdx].pt
            disparity = pt_left[0] - pt_right[0]
            if disparity < 0: raise ValueError("Negative disparity")
            pixel_u_list.append(pt_left[0])
            pixel_v_list.append(pt_left[1])
            disparity_list.append(disparity)
            depth_list.append(stereo_calib.f * stereo_calib.baseline / disparity)

        ## Output
        result_matches_dir = "./result_matches/"
        if not os.path.exists(result_matches_dir):
            os.makedirs(result_matches_dir)

        with open(f"{result_matches_dir}P3_results_{'training' if training else 'test'}_{sample_name}.txt", "a") as output_file:
            output_file.truncate(0)
            for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
                line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
                output_file.write(line + '\n')

        # Draw matches
        result_imgs_dir = "./result_imgs/"
        if not os.path.exists(result_imgs_dir):
            os.makedirs(result_imgs_dir)

        img_with_matches = cv.drawMatches(img_right, kp_right, img_left, kp_left, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(f"{result_imgs_dir}P3_{'training' if training else 'test'}_{sample_name}_matches.png", img_with_matches)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.imshow(img_with_matches)
        ax.set_title(sample_name)
        plt.show()


def test_results(part_num):
    if part_num not in [2, 3]:
        raise ValueError("Can only test for part 2 or part 3")

    tot_RMSE = 0
    samples = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    for sample in samples:
        tot_RMSE += test_depth_img(f"./result_matches/P{part_num}_results_training_{sample}.txt", f"./training/gt_depth_map/{sample}.png")
    print(f"Avg RMSE: {tot_RMSE/len(samples)}")


def test_depth_img(matches_filepath, gt_img_filepath):
    gt_img = cv.imread(gt_img_filepath, cv.IMREAD_GRAYSCALE)
    depth_diffs = []
    unchecked = 0
    with open(matches_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split()
            pix_x, pix_y, est_depth = float(line_split[1]), float(line_split[2]), float(line_split[4])
            gt_depth = gt_img[round(pix_y)][round(pix_x)]
            if(gt_depth == 0):
                unchecked += 1
                continue
            depth_diffs.append(abs(est_depth - gt_depth))
    depth_diffs = np.array(depth_diffs)
    RMSE = np.sqrt(np.sum(depth_diffs**2)/(len(depth_diffs)))
    print("-------------------------------------------------------------------------")
    print(f"Evalutaing [{matches_filepath}] against [{gt_img_filepath}]")
    print(f"Avg Diff: {np.average(depth_diffs)}")
    print(f"RMSE: {RMSE}")
    print(f"Top10 Diff: {depth_diffs[np.argsort(depth_diffs)[-10:]]}")
    print(f"Bot10 Diff: {depth_diffs[np.argsort(depth_diffs)[:10]]}")
    print(f"Estimated Correct Matches: {np.count_nonzero(depth_diffs < 3)}/{len(lines) - unchecked}")
    return RMSE


if __name__ == "__main__":
    # matplotlib.use("TkAgg")
    # q1()
    # q2(training=False)
    q3()
    test_results(part_num=3)

