"""
    Inverse rectification: take the disparity map produced by
    Foundation Stereo (in the rectified frame) and project it
    back to the original camera (NIR or SWIR) frame.
"""

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from tqdm import tqdm

from clean_code.vnir_utils import depth_utils


def inverse_rectify(args, date, horizontal_flip):
    cam_type = 'nir' if not horizontal_flip else 'swir'

    if horizontal_flip:
        R = np.load(os.path.join(args.rectified_param_dir, 'cam2_R.npy'))
        P = np.load(os.path.join(args.rectified_param_dir, 'cam2_P.npy'))
        K_orig = io.loadmat(os.path.join(args.stereo_param_dir, 'intrinsic_camera2.mat'))['K'].astype(np.float32)
        dist = io.loadmat(os.path.join(args.stereo_param_dir, 'distortion_camera2.mat'))['distortion'].astype(np.float32)
    else:
        R = np.load(os.path.join(args.rectified_param_dir, 'cam1_R.npy'))
        P = np.load(os.path.join(args.rectified_param_dir, 'cam1_P.npy'))
        K_orig = io.loadmat(os.path.join(args.stereo_param_dir, 'intrinsic_camera1.mat'))['K'].astype(np.float32)
        dist = io.loadmat(os.path.join(args.stereo_param_dir, 'distortion_camera1.mat'))['distortion'].astype(np.float32)

    Q = np.load(os.path.join(args.rectified_param_dir, 'Q.npy')).astype(np.float32)
    R = R.T
    T = np.zeros((1, 3), np.float32)

    demo_outputs = sorted(glob.glob(args.demo_output_result_dir % date, recursive=True))
    os.makedirs(args.depth_output_dir % date, exist_ok=True)

    for imfile in tqdm(demo_outputs):
        disparity = np.abs(np.load(imfile)).astype(np.float32)
        depth_map = cv2.reprojectImageTo3D(disparity, Q)

        pc, pc_vec = depth_utils.depth_map_to_point_cloud(depth_map)
        depth_utils.transform_point_cloud_to_other_camera(pc, R, T)

        uv, _ = cv2.projectPoints(pc_vec, R, T, K_orig, dist)
        depth = depth_utils.depth_from_points(
            np.stack([uv[:, 0, 0], uv[:, 0, 1]], axis=-1), pc_vec,
            disparity.shape[1], disparity.shape[0])

        depth = depth_utils.filter_rectification_noise(depth)

        np.save(os.path.join(args.depth_output_dir % date, f'{cam_type}_depth.npy'), depth)
        plt.imsave(os.path.join(args.depth_output_dir % date, f'{cam_type}_depth.png'),
                   depth[:, :, 2], vmin=300, vmax=1000, cmap='nipy_spectral')
