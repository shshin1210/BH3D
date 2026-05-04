"""
    Argparser for the clean_code VNIR pipeline.

    The original arguments are taken verbatim from
    `vnir_utils/argparser.py` so we keep the captured-data /
    calibration paths consistent.

    All arguments newly introduced for the clean_code refactor are
    grouped at the bottom of `__init__` and clearly marked with
    `### NEW (clean_code)` banners. They cover:
        * HDR generation (replaces hdr.ipynb)
        * Hyperspectral reconstruction output
        * Warping (warp_gif.py integration)
"""

import argparse
import numpy as np


class Argument:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ---------------- device ----------------
        self.parser.add_argument('--cuda_device', type=str, default="cuda:0", help="cuda device")

        # ---------------- optimization ----------------
        self.parser.add_argument('--epoch', type=int, default=600)
        self.parser.add_argument('--lr', type=float, default=0.1)
        self.parser.add_argument('--decay_step', type=int, default=590)
        self.parser.add_argument('--gamma', type=float, default=0.9)
        self.parser.add_argument('--black_level_intensity', type=float, default=0.004,
                                 help='Mask out black level noise')

        # MLP related (kept for compatibility with renderer / utils)
        self.parser.add_argument('--L_bp', type=bool, default=False)
        self.parser.add_argument('--L_mlp', type=bool, default=False)
        self.parser.add_argument('--use_model_flg', type=bool, default=False)

        # ---------------- dataloader ----------------
        self.parser.add_argument('--batch_size_train', type=int, default=100)

        # ---------------- radiometric / calibration paths ----------------
        self.parser.add_argument('--radiometric_data_dir', type=str,
                                 default='./dataset/radiometric_dataset')
        self.parser.add_argument('--bandpassfilter_data_dir', type=str,
                                 default='./calibration/radiometric_calibration/dataset/bandpass_filter')
        self.parser.add_argument('--exposure_data_dir', type=str,
                                 default='./calibration/camera_calibration/camera_exposure')

        # captured real dataset
        self.parser.add_argument('--real_dataset_dir', type=str,
                                 default='./dataset/real_dataset/%s')

        # ---------------- camera ----------------
        self.parser.add_argument('--n_angle', type=int, default=181,
                                 help='Number of galvo mirror angles (-22.5 ~ +22.5 with 0.5 step)')
        self.parser.add_argument('--cam_H', type=int, default=1024)
        self.parser.add_argument('--cam_W', type=int, default=1280)

        # ---------------- Foundation Stereo ----------------
        self.parser.add_argument('--intrinsic_file',
                                 default='./FoundationStereo/assets/K.txt', type=str)
        self.parser.add_argument('--ckpt_dir',
                                 default='./FoundationStereo/pretrained_models/23-51-11/model_best_bp2-001.pth',
                                 type=str)
        self.parser.add_argument('--scale', default=1, type=float, help='downsize factor (<=1)')
        self.parser.add_argument('--hiera', default=0, type=int)
        self.parser.add_argument('--z_far', default=10, type=float)
        self.parser.add_argument('--valid_iters', type=int, default=32)
        self.parser.add_argument('--get_pc', type=int, default=1)
        self.parser.add_argument('--remove_invisible', default=1, type=int)
        self.parser.add_argument('--denoise_cloud', type=int, default=1)
        self.parser.add_argument('--denoise_nb_points', type=int, default=30)
        self.parser.add_argument('--denoise_radius', type=float, default=0.03)

        # ---------------- rectify / inverse_rectify ----------------
        self.parser.add_argument('--x_start', type=int, default=100)
        self.parser.add_argument('--y_start', type=int, default=60)

        self.parser.add_argument('--stereo_param_dir', type=str,
                                 default='./calibration/camera_calibration/camera_params')
        self.parser.add_argument('--rectified_param_dir', type=str,
                                 default='./calibration/prism_calibration/params/rectified_parameters')
        self.parser.add_argument('--stereo_image_camera1_dir', type=str,
                                 default='./dataset/real_dataset/%s_depth/%s/*.png')
        self.parser.add_argument('--stereo_image_camera2_dir', type=str,
                                 default='./dataset/real_dataset/%s_depth/%s/*.png')

        self.parser.add_argument('--new_rectdata_dir',
                                 default='./dataset/real_dataset/%s_depth/rectified')

        self.parser.add_argument('-l', '--left_imgs',
                                 default="./dataset/real_dataset/%s_depth/rectified/%s/*.png")
        self.parser.add_argument('-r', '--right_imgs',
                                 default="./dataset/real_dataset/%s_depth/rectified/%s/*.png")

        self.parser.add_argument('--disparity_output_dir',
                                 default='./dataset/real_dataset/%s_depth/disparity')
        self.parser.add_argument('--demo_output_result_dir',
                                 default="./dataset/real_dataset/%s_depth/disparity/*.npy")
        self.parser.add_argument('--depth_output_dir',
                                 default='./dataset/real_dataset/%s_depth')

        # ---------------- prism / W-model paths ----------------
        self.parser.add_argument('--npy_dir', type=str,
                                 default='./calibration/prism_calibration/prism_dataset/npy_data')
        self.parser.add_argument('--w_model_dir', type=str,
                                 default='./calibration/prism_calibration/prism_dataset/w_model_data')

        # ---------------- depth ----------------
        self.parser.add_argument('--depth_min', type=float, default=630.0)
        self.parser.add_argument('--depth_max', type=float, default=870.0)

        # ---------------- sample points ----------------
        self.parser.add_argument('--w_range', type=int, default=75)
        self.parser.add_argument('--h_range', type=int, default=75)
        self.parser.add_argument('--step_size', type=int, default=10)

        self.parser.add_argument('--crop_x_nir', type=int, default=280)
        self.parser.add_argument('--crop_y_nir', type=int, default=170)
        self.parser.add_argument('--crop_x_swir', type=int, default=330)
        self.parser.add_argument('--crop_y_swir', type=int, default=170)

        # ---------------- crop region for optimization / rendering ----------------
        self.parser.add_argument('--crop_x_start', type=int, default=330)
        self.parser.add_argument('--crop_y_start', type=int, default=170)
        self.parser.add_argument('--crop_h', type=int, default=640)
        self.parser.add_argument('--crop_w', type=int, default=740)

        # =====================================================================
        # ### NEW (clean_code) ###
        #   Arguments below were added when refactoring main.py so it can
        #   run the full pipeline (stereo -> HDR -> recon -> warp) without
        #   relying on hdr.ipynb pre-processing.
        # =====================================================================

        # --- HDR (replaces hdr.ipynb) ---
        self.parser.add_argument('--scene_name', type=str, default='extra_scene',
                                 help='[NEW] Scene name used for the HDR / recon / warp pipeline')
        self.parser.add_argument('--hdr_data_dir', type=str,
                                 default='./dataset/hdr_dataset',
                                 help='[NEW] Directory holding multi-exposure HDR captures and where the produced HDR npys are written')
        self.parser.add_argument('--hdr_fps_samples', nargs='+', type=int, default=[3, 5],
                                 help='[NEW] FPS values of the multi-exposure captures (subdirs named %s_<fps>fps)')
        self.parser.add_argument('--hdr_invalid_intensity_ratio', type=float, default=0.01,
                                 help='[NEW] Trapezoidal weight invalid-intensity ratio for HDR fusion')
        self.parser.add_argument('--hdr_max_intensity', type=int, default=256,
                                 help='[NEW] LUT length for the HDR weight trapezoid (uint8 -> 256)')
        self.parser.add_argument('--skip_hdr', action='store_true',
                                 help='[NEW] Skip HDR generation if the npy already exists')

        # --- Hyperspectral reconstruction outputs ---
        self.parser.add_argument('--recon_output_dir', type=str, default='./result_npys',
                                 help='[NEW] Directory to dump optimized hyperspectral npys')

        # --- Warping (warp_gif.py integration) ---
        self.parser.add_argument('--warp_output_dir', type=str, default='./warpped',
                                 help='[NEW] Directory to dump warped + guided-filtered SWIR images')
        self.parser.add_argument('--warp_depth_thresh_mm', type=float, default=30.0,
                                 help='[NEW] depth1/depth2 consistency threshold in mm for occlusion gating')
        self.parser.add_argument('--warp_smooth_depth1', type=bool, default=True,
                                 help='[NEW] bilateral-smooth depth1 before warping')
        self.parser.add_argument('--warp_smooth_ksize', type=int, default=7,
                                 help='[NEW] bilateral filter kernel size for warp smoothing')
        self.parser.add_argument('--warp_smooth_sigma', type=float, default=50.0,
                                 help='[NEW] bilateral filter sigma (mm scale) for warp smoothing')
        self.parser.add_argument('--guided_r', type=int, default=12,
                                 help='[NEW] Guided filter radius')
        self.parser.add_argument('--guided_eps', type=float, default=3e-4,
                                 help='[NEW] Guided filter regularization (eps)')
        self.parser.add_argument('--guided_alpha', type=float, default=3.0,
                                 help='[NEW] Guided filter detail-transfer gain')
        self.parser.add_argument('--swir_warp_idx_start', type=int, default=17,
                                 help='[NEW] Inclusive start index of SWIR wvls to warp')
        self.parser.add_argument('--swir_warp_idx_end', type=int, default=43,
                                 help='[NEW] Exclusive end index of SWIR wvls to warp')
        self.parser.add_argument('--swir_guide_idx', type=int, default=27,
                                 help='[NEW] Index of the SWIR wvl used as guide for guided filtering')

        # --- Pipeline switches ---
        self.parser.add_argument('--cam_types', nargs='+', type=str, default=['nir', 'swir'],
                                 help='[NEW] Which cameras to run the recon for')
        self.parser.add_argument('--run_warp', action='store_true', default=True,
                                 help='[NEW] Run warping after recon')

    def parse(self):
        args = self.parser.parse_args()

        args.position = np.array(["front", "mid1", "mid2", "back"])
        args.depth_arange = np.arange(args.depth_min, args.depth_max + 1, 1)

        # bp samples
        args.wvls_samples_nir = np.array([450, 500, 550, 600, 650, 700, 750, 800, 850, 900])
        args.wvls_samples_swir = np.arange(900, 1501, 100)
        args.fps_nir = np.array([2])
        args.fps_swir = np.array([2])

        if args.L_bp:
            args.date = np.array([f"mid1_{w}nm" for w in args.wvls_samples_nir], dtype=object)
        else:
            # ### NEW (clean_code): single scene name drives the date array
            args.date = np.array([args.scene_name], dtype=object)

        args.fwhm_samples_nir = np.array([550])
        args.fwhm_samples_swir = np.array([1000])

        args.interp_wvls_nir = np.arange(450, 1001, 10)
        args.interp_wvls_swir = np.arange(450, 1701, 25)

        args.n_samples = len(args.date) * args.n_angle * args.cam_H * args.cam_W // 100
        return args
