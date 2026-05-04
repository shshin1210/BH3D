"""
    ### NEW (clean_code) ###
    Warping + guided-filter detail transfer for hyperspectral output.

    Port of `warpped/warp_gif.py` reorganized as a library module so
    main.py can call `warp_swir_to_nir` directly. The main behavioural
    changes versus the original are:
      * No reliance on `sys.path.append('../../VNIR')`; we import
        `clean_code.vnir_utils.utils` instead.
      * Hard-coded scene names removed; `args.scene_name` drives paths.
      * The main loop accepts in-memory arrays from main.py (rather
        than loading swir_<scene>.npy / nir_<scene>.npy from disk),
        but a backwards-compatible disk-loading helper is also kept.
"""

import os

import cv2
import numpy as np
from tqdm import tqdm

from guided_filter.core.filter import GuidedFilter

from clean_code.vnir_utils import utils


# --------------------------------------------------------------------------
# Bilinear sample (numpy)
# --------------------------------------------------------------------------
def sample_bilinear(img, x, y):
    H, W = img.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return (wa * img[y0c, x0c] + wb * img[y0c, x1c] +
            wc * img[y1c, x0c] + wd * img[y1c, x1c])


# --------------------------------------------------------------------------
# Depth-consistent inverse warp from cam2 (right) into cam1 (left) frame
# --------------------------------------------------------------------------
def inverse_warp_with_depth_check(
    img2, depth1, depth2,
    K1, dist1, K2, dist2, R_1to2, T_1to2,
    depth_thresh_mm=30.0,
    smooth_depth1=False, smooth_ksize=7, smooth_sigma=50.0,
):
    H, W = depth1.shape
    depth1 = depth1.astype(np.float64)
    depth2 = depth2.astype(np.float64)
    K1 = K1.astype(np.float64); K2 = K2.astype(np.float64)
    dist1 = None if dist1 is None else np.asarray(dist1, np.float64).reshape(-1)
    dist2 = None if dist2 is None else np.asarray(dist2, np.float64).reshape(-1)
    R = R_1to2.astype(np.float64)
    T = np.asarray(T_1to2, np.float64).reshape(3, 1)

    if smooth_depth1:
        d1 = depth1.copy()
        zero = d1 <= 0
        d1[zero] = 1e9
        d1 = cv2.bilateralFilter(d1.astype(np.float32), smooth_ksize, smooth_sigma, smooth_sigma)
        d1 = d1.astype(np.float64)
        d1[zero] = 0
        depth1 = d1

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv1 = np.stack([u.reshape(-1), v.reshape(-1)], axis=1).astype(np.float64)
    z1 = depth1.reshape(-1)
    valid_z1 = z1 > 0
    uv1_valid = uv1[valid_z1]
    z1_valid = z1[valid_z1]

    if dist1 is not None:
        xy1 = cv2.undistortPoints(uv1_valid.reshape(-1, 1, 2), K1, dist1, P=None).reshape(-1, 2)
        x1n, y1n = xy1[:, 0], xy1[:, 1]
    else:
        x1n = (uv1_valid[:, 0] - K1[0, 2]) / K1[0, 0]
        y1n = (uv1_valid[:, 1] - K1[1, 2]) / K1[1, 1]

    X1 = np.stack([x1n * z1_valid, y1n * z1_valid, z1_valid], axis=1)
    X2 = (R @ X1.T + T).T
    z2_pred = X2[:, 2]
    infront = z2_pred > 0
    X2 = X2[infront]; z2_pred = z2_pred[infront]
    valid_indices = np.nonzero(valid_z1)[0][infront]

    rvec0 = np.zeros((3, 1), np.float64); tvec0 = np.zeros((3, 1), np.float64)
    uv2 = cv2.projectPoints(X2, rvec0, tvec0, K2, dist2)[0].reshape(-1, 2)
    u2, v2 = uv2[:, 0], uv2[:, 1]

    inside = (u2 >= 0) & (u2 <= W - 1) & (v2 >= 0) & (v2 <= H - 1)
    u2 = u2[inside]; v2 = v2[inside]
    z2_pred = z2_pred[inside]; valid_indices = valid_indices[inside]

    z2_meas = sample_bilinear(depth2, u2, v2)
    ok = (z2_meas > 0) & (np.abs(z2_meas - z2_pred) <= depth_thresh_mm)
    u2 = u2[ok]; v2 = v2[ok]; valid_indices = valid_indices[ok]

    mapx = np.full((H * W,), -1.0, np.float32)
    mapy = np.full((H * W,), -1.0, np.float32)
    mapx[valid_indices] = u2.astype(np.float32)
    mapy[valid_indices] = v2.astype(np.float32)
    mapx = mapx.reshape(H, W); mapy = mapy.reshape(H, W)

    warped = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (mapx >= 0) & (mapy >= 0)
    return warped, valid


# --------------------------------------------------------------------------
# Guided filter detail transfer
# --------------------------------------------------------------------------
def _to_float01(u8):
    return u8.astype(np.float32) / 255.0


def _to_u8(x01):
    return np.clip(x01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def guided_detail_transfer(I_u8, p_u8, r, eps, alpha):
    I = _to_float01(I_u8)
    p = _to_float01(p_u8)
    GF = GuidedFilter(I, r, eps)
    base_p = GF.filter(p)
    base_I = GF.filter(I)
    detail_I = I - base_I
    out = np.clip(base_p + alpha * detail_I, 0, 1)
    return _to_u8(out)


# --------------------------------------------------------------------------
# Public helpers used by main.py
# --------------------------------------------------------------------------
def uncrop_recon(args, img_crop, n_wvls, cam_type):
    """
        Lift the (n_wvls, crop_h, crop_w) reconstruction back to the
        full-frame (n_wvls, H, W) so we can warp/filter at full resolution.
    """
    crop_x = args.crop_x_nir if cam_type == 'nir' else args.crop_x_swir
    full = np.zeros((n_wvls, args.cam_H, args.cam_W), dtype=np.float32)
    full[:, args.crop_y_nir:args.crop_y_nir + args.crop_h,
         crop_x:crop_x + args.crop_w] = img_crop
    return full


def warp_swir_to_nir(args, swir_unwarped, nir_unwarped):
    """
        For each SWIR wavelength index in
            [args.swir_warp_idx_start, args.swir_warp_idx_end):
          1. Detail-transfer using NIR-side guide image (same scene).
          2. Warp from SWIR-cam frame to NIR-cam frame using both
             depth maps with an occlusion gate.
        Saves PNGs into args.warp_output_dir / scene_name.

        Returns the full warped stack for SWIR.
    """
    scene = args.scene_name
    save_dir = os.path.join(args.warp_output_dir, scene)
    os.makedirs(save_dir, exist_ok=True)

    nir_depth = np.load(os.path.join(args.depth_output_dir % scene, 'nir_depth.npy'))
    swir_depth = np.load(os.path.join(args.depth_output_dir % scene, 'swir_depth.npy'))

    swir_full = uncrop_recon(args, swir_unwarped, len(args.interp_wvls_swir), 'swir')
    nir_full = uncrop_recon(args, nir_unwarped, len(args.interp_wvls_nir), 'nir')

    # NIR guide is unused here unless you want symmetric warping; keep
    # the SWIR guide as in warp_gif.py.
    swir_guide = swir_full[args.swir_guide_idx]

    K1, K2, d1, d2, R, T = utils.get_camera_parameters(args)

    swir_warped = np.zeros_like(swir_full)
    for i in tqdm(range(args.swir_warp_idx_start, args.swir_warp_idx_end),
                  desc='[Warp-swir]'):
        guide_u8 = (swir_guide * 255).astype(np.uint8)
        target_u8 = (swir_full[i] * 255).astype(np.uint8)
        gf = guided_detail_transfer(guide_u8, target_u8,
                                    r=args.guided_r, eps=args.guided_eps, alpha=args.guided_alpha)

        warped, _ = inverse_warp_with_depth_check(
            gf, nir_depth[:, :, 2], swir_depth[:, :, 2],
            K1, d1.reshape(-1), K2, d2.reshape(-1), R, T,
            depth_thresh_mm=args.warp_depth_thresh_mm,
            smooth_depth1=args.warp_smooth_depth1,
            smooth_ksize=args.warp_smooth_ksize,
            smooth_sigma=args.warp_smooth_sigma,
        )
        swir_warped[i] = warped.astype(np.float32)
        cv2.imwrite(os.path.join(save_dir, f'swir_{args.interp_wvls_swir[i]}nm.png'), warped)

    return swir_warped, nir_full
