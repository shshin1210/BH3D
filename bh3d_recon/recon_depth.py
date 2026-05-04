"""
    Depth reconstruction using Foundation Stereo.

    The clean_code variant only keeps `recon_depth_foundation` (the
    pre-trained Foundation Stereo path); the older RAFT-Stereo path
    in ../vnir_recon/recon_depth.py is dropped.
"""

import glob
import logging
import os
import sys
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as f
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

# Foundation Stereo lives at the repo root, not inside clean_code.
sys.path.append('./FoundationStereo')
sys.path.append('./FoundationStereo/cores')

from core.foundation_stereo import FoundationStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402
from Utils import set_logging_format, set_seed, vis_disparity  # noqa: E402

from clean_code.vnir_utils import inverse_rectify, rectify


class ReconDepth:
    def __init__(self, args):
        self.args = args

    def _load_image(self, path):
        img = np.array(Image.open(path)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.args.cuda_device)

    # --------------------------------------------------------------
    def recon_depth_foundation(self, date, horizontal_flip):
        """
            Run Foundation Stereo on the rectified pair and dump the
            disparity to `args.disparity_output_dir`. Then call
            inverse_rectify to project the disparity into the original
            camera frame.
        """
        cam_type = 'nir' if not horizontal_flip else 'swir'
        device = self.args.cuda_device
        print(f'[Depth] device: {device}')

        rectify.rectify(self.args, date, cam_type)

        out_dir = Path(self.args.disparity_output_dir % date)
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            set_logging_format()
            set_seed(0)
            torch.autograd.set_grad_enabled(False)

            ckpt_dir = self.args.ckpt_dir
            cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')

            allowed = {
                "left_file", "right_file", "intrinsic_file",
                "ckpt_dir", "out_dir",
                "scale", "hiera", "valid_iters", "z_far",
                "get_pc", "remove_invisible",
                "denoise_cloud", "denoise_nb_points", "denoise_radius",
                "vit_size",
            }
            for k, v in vars(self.args).items():
                if k not in allowed:
                    continue
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                cfg[k] = v
            cfg = OmegaConf.create(cfg)

            model = FoundationStereo(cfg)
            ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
            logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
            model.load_state_dict(ckpt['model'])
            model.cuda().eval()

            left_glob = sorted(glob.glob(self.args.left_imgs % (date, 'nir'), recursive=True))
            right_glob = sorted(glob.glob(self.args.right_imgs % (date, 'swir'), recursive=True))
            if horizontal_flip:
                img0 = f.hflip(Image.fromarray(imageio.imread(right_glob[0])))
                img1 = f.hflip(Image.fromarray(imageio.imread(left_glob[0])))
                img0 = np.asarray(img0)
                img1 = np.asarray(img1)
            else:
                img0 = imageio.imread(left_glob[0])
                img1 = imageio.imread(right_glob[0])

            scale = self.args.scale
            assert scale <= 1, "scale must be <=1"
            img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
            img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
            H, W = img0.shape[:2]
            img0_ori = img0.copy()

            img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0, img1 = padder.pad(img0, img1)

            with torch.cuda.amp.autocast(True):
                if not self.args.hiera:
                    disp = model.forward(img0, img1, iters=self.args.valid_iters, test_mode=True)
                else:
                    disp = model.run_hierachical(img0, img1, iters=self.args.valid_iters,
                                                 test_mode=True, small_ratio=0.5)

            disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)
            if horizontal_flip:
                disp = f.hflip(torch.from_numpy(disp).float()).cpu().numpy()

            vis = np.concatenate([img0_ori, vis_disparity(disp)], axis=1)
            imageio.imwrite(f'{out_dir}/vis.png', vis)

            np.save(os.path.join(out_dir, 'capture_0000.npy'), disp.squeeze())
            plt.imshow(disp.squeeze(), cmap='jet', vmin=0, vmax=900)
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('Disparity map (pixel)', fontdict={'fontsize': 18})
            plt.savefig(os.path.join(out_dir, 'capture_0000.png'), bbox_inches='tight', dpi=400)
            plt.clf()
            plt.close()

        inverse_rectify.inverse_rectify(self.args, date, horizontal_flip)

    # --------------------------------------------------------------
    def get_depth(self, date, flip_flg):
        """
            Returns (n_dates, H*W) array of z-depths in mm.
            If a cached `*_depth.npy` exists, it is loaded directly.
        """
        cam_type = 'nir' if not flip_flg else 'swir'
        depths = np.zeros((len(date), self.args.cam_H * self.args.cam_W), dtype=np.float32)
        for i in range(len(date)):
            cache = os.path.join(self.args.depth_output_dir % date[i], f'{cam_type}_depth.npy')
            if not os.path.exists(cache):
                self.recon_depth_foundation(date[i], horizontal_flip=flip_flg)
            depth = np.load(cache)
            depths[i] = depth.reshape(-1, 3)[:, 2]
        return depths
