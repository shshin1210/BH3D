"""
    Sample-point dataset used by the W-MLP path of the renderer.

    Trimmed port of ../vnir_sl/datatools.py with the experimental
    random-sampling variant removed.
"""

import numpy as np
from torch.utils.data import Dataset


class DataTools(Dataset):
    def __init__(self, args, cam_type, date_idx, theta_idx, depth):
        self.date_idx = date_idx
        self.theta_idx = theta_idx
        self.depth = depth.reshape(-1, args.cam_H, args.cam_W)

        if cam_type == 'nir':
            crop_x, crop_y = args.crop_x_nir, args.crop_y_nir
        else:
            crop_x, crop_y = args.crop_x_swir, args.crop_y_swir

        self.sample_pts = np.array(
            [[crop_x + w * args.step_size, crop_y + h * args.step_size]
             for h in range(args.h_range) for w in range(args.w_range)])

        self.indices = [(y, x)
                        for y in range(self.sample_pts[0, 1], self.sample_pts[-1, 1])
                        for x in range(self.sample_pts[0, 0], self.sample_pts[-1, 0])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        y, x = self.indices[idx]
        return {
            "x": x,
            "y": y,
            "Z": self.depth[self.date_idx, y, x],
            "theta": float(self.theta_idx),
            "date_idx": self.date_idx,
        }
