"""
    ### NEW (clean_code) ###
    End-to-end VNIR pipeline driver.

    This script replaces the original main.py. The old flow required
    running hdr.ipynb first to dump per-scene HDR npys, then main.py.
    Here the HDR step is integrated, so a single command runs:

        1. Stereo image preparation     (max-projection of galvo scans)
        2. HDR generation               (replaces hdr.ipynb)
        3. Foundation Stereo depth      (NIR view + SWIR view)
        4. Hyperspectral optimization   (per camera)
        5. Warping + guided detail xfer (SWIR -> NIR)

    Run from the repository root:
        python -m clean_code.main --scene_name extra_scene
"""

import os
import sys

import numpy as np

# Make the repo root importable both as a package (clean_code.*) and
# as the CWD-based `vnir_recon`/`vnir_utils` references that
# Foundation Stereo expects. We assume invocation from the repo root.
sys.path.insert(0, os.getcwd())

from clean_code.vnir_recon.recon_depth import ReconDepth
from clean_code.vnir_recon.recon_vnir import ReconVNIR
from clean_code.vnir_utils import argparser
from clean_code.vnir_utils.hdr import make_hdr_npy
from clean_code.vnir_utils.utils import (
    get_radiometric_data,
    get_wvls,
    make_stereo_images,
)
from clean_code.vnir_utils.warp import warp_swir_to_nir


def _save_recon(args, cam_type, opt_param):
    os.makedirs(args.recon_output_dir, exist_ok=True)
    out_path = os.path.join(args.recon_output_dir,
                            f'{cam_type}_{args.scene_name}.npy')
    np.save(out_path, opt_param)
    print(f'[Recon] saved -> {out_path}')


def _load_recon_if_present(args, cam_type):
    out_path = os.path.join(args.recon_output_dir,
                            f'{cam_type}_{args.scene_name}.npy')
    if os.path.exists(out_path):
        return np.load(out_path)
    return None


def main(args):
    print(f'\n=== VNIR clean pipeline | scene = "{args.scene_name}" ===\n')

    # ------------------------------------------------------------
    # 1. Stereo image preparation
    # ------------------------------------------------------------
    print('[Step 1/5] making stereo images by max-projection of galvo scans')
    make_stereo_images(args)

    recon_outputs = {}

    for cam_type in args.cam_types:
        flip_flg = (cam_type == 'swir')

        # ------------------------------------------------------------
        # 2. HDR generation (replaces hdr.ipynb)
        # ------------------------------------------------------------
        print(f'\n[Step 2/5] HDR generation for {cam_type}')
        hdr_stack = make_hdr_npy(args, cam_type)

        # ------------------------------------------------------------
        # 3. Depth reconstruction
        # ------------------------------------------------------------
        print(f'\n[Step 3/5] depth reconstruction for {cam_type}')
        recon_depth = ReconDepth(args)
        depths = recon_depth.get_depth(args.date, flip_flg)

        # ------------------------------------------------------------
        # 4. Hyperspectral optimization
        # ------------------------------------------------------------
        print(f'\n[Step 4/5] hyperspectral recon for {cam_type}')
        wvls = get_wvls(args, cam_type)
        L_lambda, _srf = get_radiometric_data(args.radiometric_data_dir, cam_type, wvls)

        recon_vnir = ReconVNIR(args, cam_type)
        opt_param = recon_vnir.optimization(L_lambda, wvls, depths, hdr_stack)

        _save_recon(args, cam_type, opt_param)
        recon_outputs[cam_type] = opt_param

    # ------------------------------------------------------------
    # 5. Warping (SWIR -> NIR view) with guided detail transfer
    # ------------------------------------------------------------
    if args.run_warp and ('nir' in recon_outputs or _load_recon_if_present(args, 'nir') is not None) \
                    and ('swir' in recon_outputs or _load_recon_if_present(args, 'swir') is not None):
        print('\n[Step 5/5] warping SWIR to NIR view')
        nir_opt = recon_outputs.get('nir', _load_recon_if_present(args, 'nir'))
        swir_opt = recon_outputs.get('swir', _load_recon_if_present(args, 'swir'))

        # opt_param shape: (n_dates, len(wvls), crop_h*crop_w). With a
        # single scene the leading dim is 1, so we collapse it.
        nir_unwarped = nir_opt.reshape(-1, len(args.interp_wvls_nir),
                                       args.crop_h, args.crop_w)[0]
        swir_unwarped = swir_opt.reshape(-1, len(args.interp_wvls_swir),
                                         args.crop_h, args.crop_w)[0]

        warp_swir_to_nir(args, swir_unwarped, nir_unwarped)
    else:
        print('[Step 5/5] skipping warping — need both NIR and SWIR recon outputs.')

    print('\n=== pipeline finished ===')


if __name__ == '__main__':
    argument = argparser.Argument()
    args = argument.parse()
    main(args)
