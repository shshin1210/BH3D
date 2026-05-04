import glob
import os
import time

import cv2
import numpy as np
import scipy.io as io
from tqdm import tqdm


def rectify(args, date, cam_type):
    base = args.stereo_param_dir

    K1 = io.loadmat(os.path.join(os.getcwd(), base, 'intrinsic_camera1.mat'))['K'].astype(np.float64)
    K2 = io.loadmat(os.path.join(os.getcwd(), base, 'intrinsic_camera2.mat'))['K'].astype(np.float64)
    d1 = io.loadmat(os.path.join(os.getcwd(), base, 'distortion_camera1.mat'))['distortion'].astype(np.float64)
    d2 = io.loadmat(os.path.join(os.getcwd(), base, 'distortion_camera2.mat'))['distortion'].astype(np.float64)
    R = io.loadmat(os.path.join(os.getcwd(), base, 'rotation.mat'))['R'].astype(np.float64)
    T = io.loadmat(os.path.join(os.getcwd(), base, 'translation.mat'))['T'].astype(np.float64)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, d1, K2, d2, (args.cam_H, args.cam_W), R, T.T)

    rect_param_dir = os.path.join(os.getcwd(), args.rectified_param_dir)
    os.makedirs(rect_param_dir, exist_ok=True)
    np.save(os.path.join(rect_param_dir, 'cam1_R.npy'), R1)
    np.save(os.path.join(rect_param_dir, 'cam2_R.npy'), R2)
    np.save(os.path.join(rect_param_dir, 'cam1_P.npy'), P1)
    np.save(os.path.join(rect_param_dir, 'cam2_P.npy'), P2)
    np.save(os.path.join(rect_param_dir, 'Q.npy'), Q)

    left_images = sorted(glob.glob(args.stereo_image_camera1_dir % (date, 'nir'), recursive=True))
    left_images = [f for f in left_images if 'black.png' not in f]
    right_images = sorted(glob.glob(args.stereo_image_camera2_dir % (date, 'swir'), recursive=True))
    right_images = [f for f in right_images if 'black.png' not in f]

    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, (args.cam_W, args.cam_H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, (args.cam_W, args.cam_H), cv2.CV_32FC1)

    out_root = os.path.join(os.getcwd(), args.new_rectdata_dir % date)
    os.makedirs(os.path.join(out_root, 'nir'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'swir'), exist_ok=True)

    print(f'rectifying {date} ({cam_type}) ======>')
    for f1, f2 in tqdm(list(zip(left_images, right_images))):
        im1 = cv2.imread(f1)
        im2 = cv2.imread(f2)
        if (im1.shape[0], im1.shape[1]) != (args.cam_H, args.cam_W):
            im1 = cv2.resize(im1, (args.cam_W, args.cam_H))
        r1 = cv2.remap(im1, map1x, map1y, cv2.INTER_LINEAR)
        r2 = cv2.remap(im2, map2x, map2y, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_root, 'nir', os.path.basename(f1)), r1)
        cv2.imwrite(os.path.join(out_root, 'swir', os.path.basename(f2)), r2)

    time.sleep(1)
