"""
    Depth helper functions used by inverse_rectify.

    Verbatim port from ../vnir_utils/depth_utils.py.
"""

import numpy as np
import open3d as o3d


def depth_map_to_point_cloud(depth_map, intrinsic=None):
    """
        depth_map : (H, W, 3) where last dim is unprojected (x, y, z).
        Returns (PointCloud, ndarray points).
    """
    if intrinsic is None:
        mask = (depth_map[:, :, 2] >= -np.inf) & (depth_map[:, :, 2] <= np.inf)
        points = depth_map[mask]
    else:
        fx, fy = intrinsic[0][0], intrinsic[1][1]
        cx, cy = intrinsic[0][2], intrinsic[1][2]
        rows, cols = depth_map.shape
        xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
        zz = depth_map
        x = (xx - cx) * zz / fx
        y = (yy - cy) * zz / fy
        points = np.vstack((x.flatten(), y.flatten(), zz.flatten())).T

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc, points


def transform_point_cloud_to_other_camera(point_cloud, R, T):
    point_cloud.transform(np.vstack([np.hstack((R, T.reshape(-1, 1))), [0, 0, 0, 1]]))
    return point_cloud


def depth_from_points(points_uv, points_xyz, W, H):
    depth_map = np.zeros((H, W, 3))
    u, v = points_uv[:, 0], points_uv[:, 1]
    inside = (0 <= u) & (u <= W - 1) & (0 <= v) & (v <= H - 1)
    depth_map[np.round(v[inside]).astype(int), np.round(u[inside]).astype(int)] = points_xyz[inside]
    return depth_map


def filter_rectification_noise(img):
    """
        Fill 0-depth holes by averaging adjacent valid pixels.
    """
    H, W = img.shape[:2]
    out = np.array(img)
    zero_idx = np.where(img[:, :, 2] == 0)
    for (i, j) in zip(zero_idx[0], zero_idx[1]):
        adj = [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j - 1)]
        s, c = 0, 0
        for ai, aj in adj:
            if ai >= H or aj >= W:
                continue
            v = img[ai, aj]
            if v[2] != 0:
                s += v
                c += 1
        if c > 0:
            out[i, j] = s / c
    return out
