"""
    Hyperspectral reflectance optimization.

    The clean_code variant takes the HDR stack as an in-memory tensor
    instead of reading `<hdr_data_dir>/%s_hdr_%s.npy` from disk. This
    lets `main.py` produce the HDR stack and immediately feed it into
    the optimizer without an intermediate save/load.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from clean_code.vnir_sl.render import Renderer
from clean_code.vnir_utils.utils import mask


class ReconVNIR:
    """Optimize per-pixel hyperspectral reflectance H(lambda)."""

    def __init__(self, args, cam_type):
        self.args = args
        self.device = args.cuda_device
        self.cam_type = cam_type

    # --------------------------------------------------------------
    def crop(self, data):
        x_tl, y_tl = self.args.crop_x_start, self.args.crop_y_start
        h, w = self.args.crop_h, self.args.crop_w
        return data[y_tl:y_tl + h, x_tl:x_tl + w]

    def tv_loss_fn_crop(self, data, date_idx, L_lambda):
        d = data[date_idx].reshape(-1, self.args.crop_h, self.args.crop_w)
        Lr = L_lambda.unsqueeze(dim=-1)

        y_dL = (abs(d[:, :-1] - d[:, 1:]) * (1 / Lr)).sum() / (self.args.crop_h * self.args.crop_w)
        x_dL = (abs(d[:, :, :-1] - d[:, :, 1:]) * (1 / Lr)).sum() / (self.args.crop_h * self.args.crop_w)
        wvl_dL = (abs(d[:-1] - d[1:]) * (1 / Lr[1:])).sum() / (self.args.crop_h * self.args.crop_w)

        return x_dL + y_dL + wvl_dL

    # --------------------------------------------------------------
    def optimization(self, L_lambda, wvls, depths, hdr_stack):
        """
            Args:
                L_lambda  : (len(wvls),) numpy
                wvls      : (len(wvls),) numpy
                depths    : (n_dates, H*W) numpy in mm
                hdr_stack : (n_angle, H, W) HDR stack (uint8-scaled)
                            -> normalized to [0, 1] inside this function

            Returns the optimized H_lambda as a numpy array of shape
            (n_dates, len(wvls), crop_h*crop_w).
        """
        print(f'[Recon-{self.cam_type}] start hyperspectral optimization')

        # ---- optimization variable ----
        init = torch.ones((len(self.args.date), len(wvls),
                           self.args.crop_h * self.args.crop_w)) / 2
        _opt = torch.tensor(init, dtype=torch.float, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([_opt], lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.decay_step, gamma=self.args.gamma)

        # ---- inputs to device ----
        gt = hdr_stack[np.newaxis, ...] / 255.0
        gt_dev = torch.tensor(gt, device=self.device, dtype=torch.float32)
        mask_dev = torch.tensor(mask(self.args, self.cam_type),
                                dtype=torch.float32, device=self.device)

        renderer = Renderer(self.args, self.args.date, self.cam_type, wvls, depths, mask=mask_dev)

        const = 1.2 if self.cam_type == 'nir' else 1.4
        L_dev = torch.tensor(L_lambda * const, device=self.device)

        chunk_start, chunk_end = 10, 180
        batch_angle_size = 6
        loss_fn = torch.nn.MSELoss()

        for epoch in tqdm(range(self.args.epoch)):
            for date_idx in range(len(self.args.date)):
                for batch_start in range(chunk_start, chunk_end, batch_angle_size):
                    batch_end = min(batch_start + batch_angle_size, chunk_end)

                    optimizer.zero_grad()
                    batch_loss = 0.0
                    H_lambda = torch.sigmoid(_opt)

                    for angle_idx in range(batch_start, batch_end):
                        sim = renderer.gaussian_render_crop(
                            L_lambda=L_dev,
                            H_lambda=H_lambda[date_idx],
                            date_idx=date_idx,
                            angle_idx=angle_idx,
                        )
                        gt_crop = self.crop(gt_dev[date_idx, angle_idx])
                        m_crop = self.crop(mask_dev)
                        batch_loss += loss_fn(sim, gt_crop * m_crop) * 5

                    batch_loss = batch_loss + self.tv_loss_fn_crop(H_lambda, date_idx, L_dev) * (1e-6 * 6)
                    batch_loss.backward()
                    optimizer.step()
                    scheduler.step()

        print(f'[Recon-{self.cam_type}] optimization done')
        return torch.sigmoid(_opt).detach().cpu().numpy()
