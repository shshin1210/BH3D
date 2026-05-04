"""
    Spectrally-resolved renderer.

    Only the renderer paths used by the clean_code pipeline are kept:
        - constructor pre-loads the depth-conditioned (mean, std) maps
        - gaussian_render_crop produces the simulated cropped image
          for one (date, angle) pair given (L_lambda, H_lambda)

    Anything related to bandpass-filter rendering, naive rendering, or
    the W-MLP code path is dropped because the clean main.py does not
    exercise them.
"""

import numpy as np
import torch
from tqdm import tqdm

from clean_code.vnir_utils.utils import scene_dependent_model_sharded_interp


class Renderer:
    def __init__(self, args, date, cam_type, wvls, depth, mask=None):
        self.args = args
        self.wvls = wvls
        self.cam_type = cam_type
        self.date = date
        self.depth = depth
        self.mask = mask

        print(f'[Renderer] building scene-dependent W model for {cam_type}...')
        scene_mean = []
        scene_std = []
        for d in tqdm(range(len(date))):
            scene_mean.append(scene_dependent_model_sharded_interp(
                args, len(wvls), depth[d], cam_type, kind="mean"))
            scene_std.append(scene_dependent_model_sharded_interp(
                args, len(wvls), depth[d], cam_type, kind="std"))

        scene_mean = np.stack([m.reshape(len(wvls), -1) for m in scene_mean], axis=0)
        scene_std = np.stack([s.reshape(len(wvls), -1) for s in scene_std], axis=0)

        self.mean = torch.tensor(scene_mean, dtype=torch.float32, device=args.cuda_device)
        self.std = torch.tensor(scene_std, dtype=torch.float32, device=args.cuda_device)

    # ------------------------------------------------------------
    def _crop(self, data):
        x_tl, y_tl = self.args.crop_x_start, self.args.crop_y_start
        h, w = self.args.crop_h, self.args.crop_w
        return data[:, y_tl:y_tl + h, x_tl:x_tl + w]

    def _valid_mask(self, data, mask):
        valid = (mask.reshape(-1) == 1)
        data_valid = data[:, valid]
        data_mean = data_valid.mean(axis=1)
        data_mean_exp = data_mean.unsqueeze(1).expand(-1, data.shape[1])
        return torch.where(mask.reshape(-1).unsqueeze(0) == 1, data_mean_exp, data)

    # ------------------------------------------------------------
    def gaussian_render_crop(self, L_lambda, H_lambda, date_idx, angle_idx):
        """
            Args
            - L_lambda : (len(wvls),) radiometric weight
            - H_lambda : (len(wvls), crop_h*crop_w) hyperspectral reflectance
            - date_idx : index into self.date
            - angle_idx: galvo angle index (int)

            Returns rendered (crop_h, crop_w) intensity image.
        """
        eps = 1e-12
        std = self.std[date_idx].clamp_min(eps)
        std = self._valid_mask(std, self.mask)

        std_crop = self._crop(std.reshape(len(self.wvls), self.args.cam_H, self.args.cam_W))
        mean_crop = self._crop(self.mean[date_idx].reshape(
            len(self.wvls), self.args.cam_H, self.args.cam_W))
        std_crop = std_crop.reshape(len(self.wvls), -1)
        mean_crop = mean_crop.reshape(len(self.wvls), -1)

        angle = torch.tensor(angle_idx, device=self.args.cuda_device)
        rendered_hyp = torch.exp(-((angle - mean_crop) ** 2) / (2.0 * std_crop * std_crop))

        depth_crop = self._crop(self.depth.reshape(1, self.args.cam_H, self.args.cam_W))
        depth_dev = torch.tensor(depth_crop.reshape(-1),
                                 dtype=torch.float32,
                                 device=self.args.cuda_device) / self.args.depth_max

        w_lp = L_lambda * H_lambda
        rendered_mono = (rendered_hyp * w_lp).sum(dim=0) * (1.0 / (depth_dev ** 2))
        return rendered_mono.reshape(self.args.crop_h, self.args.crop_w)
