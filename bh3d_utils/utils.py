"""
    Common utilities used by the clean_code pipeline.

    These are trimmed down versions of the originals in ../vnir_utils/utils.py.
    Visualization / debugging helpers that were only used inside notebooks
    or rebuttal scripts have been intentionally removed for clarity.
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
import scipy.io as io
from scipy.io import loadmat


# --------------------------------------------------------------------------
# Radiometric data
# --------------------------------------------------------------------------
def get_radiometric_data(dir, cam, wvls):
    """
        Load pre-computed L_lambda and SRF for a camera.
        See ../vnir_utils/utils.py for the original docstring.
    """
    sls301L_power = np.load(os.path.join(dir, 'sls302L_power.npy'))
    nf2_prism = np.load(os.path.join(dir, 'nf2_prism.npy'), allow_pickle=True)
    nbk7_trans = np.load(os.path.join(dir, 'nbk7_trans.npy'), allow_pickle=True)
    galvo_reflectance = np.load(os.path.join(dir, 'galvo_y_interp.npy'), allow_pickle=True)
    lens_trans = np.load(os.path.join(dir, 'swir_lens_y_interp.npy'), allow_pickle=True)

    if cam == 'nir':
        srf = np.load(os.path.join(dir, 'nir_srf.npy'), allow_pickle=True)
        L_lambda = np.load(os.path.join(dir, 'final_nir_srf6.npy'))
    else:
        srf = np.load(os.path.join(dir, 'swir_srf_y_interp.npy'), allow_pickle=True)
        L_lambda = np.load(os.path.join(dir, 'final_swir_srf6.npy'))

    _ = sls301L_power * nf2_prism * nbk7_trans * lens_trans * galvo_reflectance  # not used directly
    return L_lambda.astype(np.float32), srf


def get_wvls(args, cam_type):
    if cam_type == 'nir':
        return args.interp_wvls_nir
    return args.interp_wvls_swir


# --------------------------------------------------------------------------
# Depth-conditioned scene model
# --------------------------------------------------------------------------
def scene_dependent_model_sharded_interp(args, wvl_num, depth, cam_type, kind="mean"):
    """
        Bring per-depth slices of the prism dispersion model and
        linearly interpolate them at every pixel's depth.

        Returns (wvl_num, H, W).
    """
    H, W = args.cam_H, args.cam_W
    N = H * W

    depth_flat = np.asarray(depth, dtype=np.float32).reshape(-1)
    depth_clamped = np.clip(depth_flat, args.depth_min, args.depth_max)

    depth_arange = np.asarray(args.depth_arange, dtype=np.float32)
    step = depth_arange[1] - depth_arange[0]
    base = depth_arange[0]

    idx_float = (depth_clamped - base) / step
    idx_float = np.clip(idx_float, 0.0, len(depth_arange) - 1.0)

    idx_low = np.floor(idx_float).astype(np.int64)
    idx_high = np.clip(idx_low + 1, 0, len(depth_arange) - 1)
    alpha = (idx_float - idx_low).astype(np.float32)

    slice_dir = os.path.join(args.w_model_dir, f"depth_slices_{cam_type}")
    needed = np.unique(np.concatenate([idx_low, idx_high]))
    slices = {}
    for di in needed:
        fpath = os.path.join(slice_dir, f"{kind}_depth_{di:04d}.npy")
        slices[di] = np.load(fpath).reshape(-1, N)

    out = np.zeros((wvl_num, N), dtype=np.float32)
    for li in np.unique(idx_low):
        mask_all = (idx_low == li)
        if not np.any(mask_all):
            continue
        mask_same = mask_all & (idx_high == li)
        if np.any(mask_same):
            out[:, mask_same] = slices[li][:, mask_same]
        mask_interp = mask_all & (idx_high != li)
        if np.any(mask_interp):
            hi = min(li + 1, len(depth_arange) - 1)
            a = alpha[mask_interp][None, :]
            out[:, mask_interp] = (1.0 - a) * slices[li][:, mask_interp] + a * slices[hi][:, mask_interp]

    return out.reshape(wvl_num, H, W)


# --------------------------------------------------------------------------
# Bandpass / exposure
# --------------------------------------------------------------------------
def bandpassfilter(args, cam_type, wvls):
    if cam_type == 'nir':
        wvls_samples = args.fwhm_samples_nir
    else:
        wvls_samples = args.fwhm_samples_swir

    bp_filters = np.zeros((len(wvls_samples), len(wvls)), dtype=np.float32)
    for wvl_idx, wvl in enumerate(wvls_samples):
        suffix = '10' if wvl < 1300 else '12'
        bp_path = os.path.join(args.bandpassfilter_data_dir, f'FBH{wvl}-{suffix}.xlsx')
        bp_xls = pd.read_excel(bp_path)
        for i in range(len(wvls)):
            try:
                row_idx = np.where(wvls[i] == bp_xls['Wavelength (nm)'])[0][0]
                bp_filters[wvl_idx, i] = bp_xls['% Transmission'][int(row_idx)]
            except Exception:
                bp_filters[wvl_idx, i] = 0
    return bp_filters / 100.


def fps_to_exposure(fps):
    return (1.0 / fps) * 1e6 - (1.0 / fps) * 1e5


def exposure(args, cam_type):
    param = loadmat(os.path.join(os.getcwd(), args.exposure_data_dir, f'{cam_type}_param.mat'))
    if cam_type == 'nir':
        ex = fps_to_exposure(args.fps_nir)
    else:
        ex = fps_to_exposure(args.fps_swir)
    fitted = param['p'][0][0] * ex + param['p'][0][1]
    return fitted / fitted.max()


# --------------------------------------------------------------------------
# Sample mask (defines the optimization region inside the camera image)
# --------------------------------------------------------------------------
def mask(args, cam_type):
    if cam_type == 'nir':
        sample_pts = np.array([[args.crop_x_nir + w * args.step_size,
                                args.crop_y_nir + h * args.step_size]
                               for h in range(args.h_range) for w in range(args.w_range)])
    else:
        sample_pts = np.array([[args.crop_x_swir + w * args.step_size,
                                args.crop_y_swir + h * args.step_size]
                               for h in range(args.h_range) for w in range(args.w_range)])

    m = np.zeros((args.cam_H, args.cam_W), dtype=np.float32)
    x_start, x_end = sample_pts[0, 0], sample_pts[-1, 0]
    y_start, y_end = sample_pts[0, 1], sample_pts[-1, 1]
    m[y_start:y_end, x_start:x_end] = 1
    return m


# --------------------------------------------------------------------------
# Stereo image preparation (max projection over galvo angles)
# --------------------------------------------------------------------------
def make_stereo_images(args):
    """
        Build a single sharp stereo image per camera by taking the max
        of all galvo-scanned captures. Output goes into
        `<real_dataset>/<scene>_depth/<cam>/capture_0000.png`.
    """
    for cam_type in np.array(['nir', 'swir']):
        imgs = np.zeros((args.n_angle, args.cam_H, args.cam_W))
        for i in range(args.n_angle):
            path = os.path.join(args.real_dataset_dir % args.date[0],
                                f'{cam_type}/capture_{i:04d}.png')
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if cam_type == 'swir':
                im = cv2.medianBlur(im, 5)
            imgs[i] = im

        led_max = np.max(imgs, axis=0)
        led_max = (led_max / led_max.max() * 255).astype(np.uint8)

        save_dir = os.path.join(args.real_dataset_dir % args.date[0] + '_depth', cam_type)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'capture_0000.png'), led_max)


# --------------------------------------------------------------------------
# Stereo extrinsic / intrinsic loader (used by warping)
# --------------------------------------------------------------------------
def get_camera_parameters(args):
    base = args.stereo_param_dir
    K1 = io.loadmat(os.path.join(base, 'intrinsic_camera1.mat'))['K'].astype(np.float64)
    K2 = io.loadmat(os.path.join(base, 'intrinsic_camera2.mat'))['K'].astype(np.float64)
    d1 = io.loadmat(os.path.join(base, 'distortion_camera1.mat'))['distortion'].astype(np.float64)
    d2 = io.loadmat(os.path.join(base, 'distortion_camera2.mat'))['distortion'].astype(np.float64)
    R = io.loadmat(os.path.join(base, 'rotation.mat'))['R'].astype(np.float64)
    T = io.loadmat(os.path.join(base, 'translation.mat'))['T'].astype(np.float64)
    return K1, K2, d1, d2, R, T
