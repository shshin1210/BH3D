"""
    ### NEW (clean_code) ###
    HDR generation module — replaces `hdr.ipynb`.

    For a given scene + camera, multi-exposure captures are stored under
        <hdr_data_dir>/<scene>_<fps>fps/<cam_type>/capture_XXXX.png
    where capture_0000.png is the black frame.

    For each angle index i in [0, n_angle):
        1. Read the LDR image at every fps.
        2. Subtract the corresponding black frame (capture_0000.png).
        3. Fuse using a trapezoidal weight in (LDR space) and a
           ramp weight in (black-removed LDR space). The final weight
           is the elementwise minimum of the two.
        4. Normalize each capture by its exposure (in ms) and combine.

    Output: an HDR stack of shape (n_angle, H, W) saved as
        <hdr_data_dir>/<cam_type>_hdr_<scene>.npy
"""

import os

import cv2
import numpy as np


def _safe_subtract(a, b):
    """uint8-safe subtraction returning a.dtype."""
    diff = np.where(a > b, a - b, 0)
    diff = np.clip(np.round(diff), 0, 255).astype(np.uint8)
    return diff.astype(a.dtype)


def _build_weight_trapezoid(max_intensity, invalid_ratio):
    """Trapezoidal weight LUT used for the LDR (with-black) image."""
    w = np.zeros(max_intensity, dtype=np.float32)
    intv = float(max_intensity) * invalid_ratio
    for i in range(max_intensity):
        if i < intv:
            w[i] = 0
        elif i < intv * 2:
            w[i] = (i - intv) / intv
        elif i < max_intensity - intv * 2:
            w[i] = 1
        elif i < max_intensity - intv:
            w[i] = (max_intensity - intv - i) / intv
        else:
            w[i] = 0
    return w


def _build_weight_trapezoid_bgrm(max_intensity, invalid_ratio):
    """Ramp-up weight LUT used for the black-removed image (no top falloff)."""
    w = np.zeros(max_intensity, dtype=np.float32)
    intv = float(max_intensity) * invalid_ratio
    for i in range(max_intensity):
        if i < intv:
            w[i] = 0
        elif i < intv * 2:
            w[i] = (i - intv) / intv
        else:
            w[i] = 1
    return w


def _make_hdr(ldr_imgs, ldr_imgs_bgrm, weight_trap, weight_trap_bgrm, exposure_w):
    """Fuse multi-exposure LDR pairs into a single HDR frame."""
    weighted = np.array([weight_trap[img] for img in ldr_imgs], dtype=np.float32)
    weighted_bgrm = np.array([weight_trap_bgrm[img] for img in ldr_imgs_bgrm], dtype=np.float32)

    w_final = np.minimum(weighted, weighted_bgrm)

    radiance = np.array([w_final[i] * (ldr_imgs_bgrm[i].astype(np.float32) / exposure_w[i])
                         for i in range(len(ldr_imgs_bgrm))], dtype=np.float32)

    w_sum = w_final.sum(axis=0)
    r_sum = radiance.sum(axis=0)

    invalid = (w_sum == 0)
    w_sum[invalid] = 1.0
    r_sum[invalid] = 0.0

    return r_sum / w_sum


def make_hdr_npy(args, cam_type):
    """
        Build the HDR stack for `cam_type` of `args.scene_name`.

        Returns the hdr stack (n_angle, H, W) as a float32 array and
        also writes it to disk so subsequent runs can be skipped.
    """
    scene = args.scene_name
    out_path = os.path.join(args.hdr_data_dir, f'{cam_type}_hdr_{scene}.npy')

    if args.skip_hdr and os.path.exists(out_path):
        print(f'[HDR] {cam_type}: cached -> {out_path}')
        return np.load(out_path)

    fps_samples = np.array(args.hdr_fps_samples)

    ex_time_us = (1.0 / fps_samples) * 1e6 - (1.0 / fps_samples) * 1e5
    ex_time_ms = ex_time_us / 1e3
    ex_min = ex_time_ms.min()
    exposure_w = ex_time_ms / ex_min

    weight_trap = _build_weight_trapezoid(args.hdr_max_intensity, args.hdr_invalid_intensity_ratio)
    weight_trap_bgrm = _build_weight_trapezoid_bgrm(args.hdr_max_intensity, args.hdr_invalid_intensity_ratio)

    data_dir_fmt = os.path.join(args.hdr_data_dir, '%s_%dfps', '%s')

    hdr_imgs = []
    print(f'[HDR] generating {cam_type} hdr stack for scene "{scene}"')
    for i in range(args.n_angle):
        ldr = np.array(
            [cv2.imread(os.path.join(data_dir_fmt % (scene, k, cam_type),
                                     f'capture_{i:04d}.png'), cv2.IMREAD_GRAYSCALE)
             for k in fps_samples], dtype=np.uint8)
        ldr_black = np.array(
            [cv2.imread(os.path.join(data_dir_fmt % (scene, k, cam_type),
                                     'capture_0000.png'), cv2.IMREAD_GRAYSCALE)
             for k in fps_samples], dtype=np.uint8)

        ldr_bgrm = np.clip(_safe_subtract(ldr, ldr_black), 0, 2 ** 8).astype(np.uint8)
        hdr_imgs.append(_make_hdr(ldr, ldr_bgrm, weight_trap, weight_trap_bgrm, exposure_w))

        if i % 20 == 0:
            print(f'  [HDR] {cam_type}: {i:03d}/{args.n_angle} done')

    hdr_imgs = np.stack(hdr_imgs, axis=0).astype(np.float32)
    os.makedirs(args.hdr_data_dir, exist_ok=True)
    np.save(out_path, hdr_imgs)
    print(f'[HDR] saved -> {out_path}')
    return hdr_imgs
