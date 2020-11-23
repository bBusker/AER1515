#!/usr/env/bin python3

import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib
import numpy as np
import os
# matplotlib.use('TkAgg')


from utils import *


'''
Starter code for loading files, calibration data, and transformations
'''

# File paths
calib_dir = os.path.abspath('./data/calib')
image_dir = os.path.abspath('./data/image')
lidar_dir = os.path.abspath('./data/velodyne')
sample = '000000'

# Load the image
image_path = os.path.join(image_dir, sample + '.png')
image = img.imread(image_path)

# Load the LiDAR points
lidar_path = os.path.join(lidar_dir, sample + '.bin')
lidar_points = load_velo_points(lidar_path)

# Load the body to camera and body to LiDAR transforms
body_to_lidar_calib_path = os.path.join(calib_dir, 'calib_imu_to_velo.txt')
T_lidar_body = load_calib_rigid(body_to_lidar_calib_path)

# Load the camera calibration data
# Remember that when using the calibration data, there are 4 cameras with IDs
# 0 to 3. We will only consider images from camera 2.
lidar_to_cam_calib_path = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
cam_to_cam_calib_path = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
cam_calib = load_calib_cam_to_cam(lidar_to_cam_calib_path, cam_to_cam_calib_path)
intrinsics = cam_calib['K_cam2']
T_cam2_lidar = cam_calib['T_cam2_velo']

'''
For you to complete:
'''
# Part 1: Convert LiDAR points from LiDAR to body frame (for depths)
# Note that the LiDAR data is in the format (x, y, z, r) where x, y, and z are
# distances in metres and r is a reflectance value for the point which can be
# ignored. x is forward, y is left, and z is up. Depth can be calculated using
# d^2 = x^2 + y^2 + z^2
lidar_points_hom = lidar_points
lidar_points_hom[:, 3] = 1
lidar_points_hom_B = np.linalg.inv(T_lidar_body) @ lidar_points_hom.T
depths = []
for i in range(lidar_points_hom_B.shape[1]):
    depths.append(np.sqrt(lidar_points_hom_B[0][i]**2 + lidar_points_hom_B[1][i]**2 + lidar_points_hom_B[2][i]**2))
lidar_depths = np.array(depths)

# Part 2: Convert LiDAR points from LiDAR to camera 2 frame
lidar_points_hom_C2 = T_cam2_lidar @ lidar_points_hom.T

# Part 3: Project the points from the camera 2 frame to the image plane. You
# may assume no lens distortion in the image. Remember to filter out points
# where the projection does not lie within the image field, which is 1242x375.
lidar_points_C2 = lidar_points_hom_C2[0:3, :]
lidar_points_norm_C2 = lidar_points_C2 / lidar_points_C2[2, :]
lidar_points_pix_C2 = intrinsics @ lidar_points_norm_C2
lidar_points_pix_loc = lidar_points_pix_C2[0:2, :]

# Part 4: Overlay the points on the image with the appropriate depth values.
# Use a colormap to show the difference between points' depths and remember to
# include a colorbar.
depth_img = np.zeros(image.shape[0:2])
scatter_arr = []
scatter_depths = []
img_height = image.shape[0]
img_width = image.shape[1]
count = 0
for i in range(lidar_points_pix_loc.shape[1]):
    curr_pt = lidar_points_pix_loc[0:2, i]
    if curr_pt[0] >= img_width-.5 or curr_pt[0] < 0 or curr_pt[1] >= img_height-.5 or curr_pt[1] < 0:
        continue
    # depth_img[int(round(curr_pt[1]))][int(round(curr_pt[0]))] = depths[i]
    scatter_arr.append(curr_pt)
    scatter_depths.append(depths[i])
    count += 1
print(count)
# depth_img_masked = np.ma.masked_where(depth_img == 0, depth_img)

scatter_arr = np.array(scatter_arr)
scatter_depths = np.array(scatter_depths)
print(scatter_arr.shape)

plt.figure(figsize=(20, 5))
plt.imshow(image)
# plt.imshow(depth_img_masked)
plt.scatter(scatter_arr[:, 0], scatter_arr[:, 1], c=scatter_depths, s=0.5)
plt.colorbar()
plt.show()
