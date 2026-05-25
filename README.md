# Broadband Hyperspectral 3D Imaging using Dispersed Structured Light

[BH3D](https://shshin1210.github.io/BH3D/) (Broadband Hyperspectral 3D Imaging using Dispersed Structured Light) reconstructs depth and both spectral information : Visible to SWIR (450nm ~ 1500nm) spectral ranges at 20nm (visible range) 25nm (SWIR range) interval.

## Image system configuation
<img width="1044" height="326" alt="image" src="https://github.com/user-attachments/assets/989b6e74-ac4d-46e2-a4a4-24ec480fadb0" />

**This is the BH3D imaging system configuration, please refer to the Supplementary Document for specific experimental prototype.**

## BH3D pipeline overview

`main.py` runs five steps in order. Steps 2–4 are run
once per camera (`nir`, `swir`); step 1 happens for both cameras
together; step 5 is run once at the end.

| # | Step | Module | What it does |
|---|------|--------|--------------|
| 1 | Stereo image preparation | `vnir_utils/utils.py::make_stereo_images` | Max-projects galvo-scanned captures into a single sharp image per camera and writes `<scene>_depth/<cam>/capture_0000.png`. |
| 2 | HDR generation | `vnir_utils/hdr.py::make_hdr_npy` | For each angle index, fuses the multi-exposure captures (under `<hdr_data_dir>/<scene>_<fps>fps/<cam>`) using a trapezoid weight + black-frame subtraction and writes `<hdr_data_dir>/<cam>_hdr_<scene>.npy`. |
| 3 | Depth reconstruction | `vnir_recon/recon_depth.py` | Rectifies the stereo pair, runs Foundation Stereo, then inverse-rectifies the disparity into the original camera frame. Caches `<scene>_depth/<cam>_depth.npy`. |
| 4 | Hyperspectral recon | `vnir_recon/recon_vnir.py` | Adam-optimizes per-pixel hyperspectral reflectance against the HDR stack (in-memory — no disk round-trip). Saves `<recon_output_dir>/<cam>_<scene>.npy`. |
| 5 | Warping | `vnir_utils/warp.py` | Detail-transfers using a guided filter on the SWIR side, then warps each SWIR wavelength into the NIR camera view using both depth maps + an occlusion gate. PNGs go to `<warp_output_dir>/<scene>/`. |


## BH3D layout

```
clean_code/
├── main.py
├── README.md
├── dataset/
├── calibration/
├── bh3d_utils/
│   ├── argparser.py       
│   ├── hdr.py          
│   ├── warp.py             
│   ├── utils.py
│   ├── rectify.py
│   ├── inverse_rectify.py
│   └── depth_utils.py
├── bh3d_sl/
│   ├── render.py           ← simulated rendering (gaussian_render_crop)
│   └── datatools.py
├── bh3d_recon/
    ├── recon_depth.py      ← Foundation Stereo wrapper
    └── recon_vnir.py       ← in-memory HDR variant
├── FoundationStereo/
└── guided_filter/
```

External dependencies (must remain at the repo root and were not
modified):

* `FoundationStereo/` — pretrained stereo model
* `guided_filter/` — guided-filter implementation used by sharpening
* `calibration/` — radiometric, prism, and stereo calibration data
* `dataset/` — captured frames, radiometric data, and HDR raw captures

We provide an expample calibration parameters and datsets in our [BH3D Calibration Parameters](https://drive.google.com/drive/u/0/folders/128apzV3A4GjllRUOafMHEM0yIY_EZvLf).
Please refer to our Main paper and Supplementary Document for the details of data-driven Gaussian Model.

## Argparser

All new arguments live in `bh3d_utils/argparser.py`. They
are introduced under banners that read

They cover:

* HDR: `--scene_name`, `--hdr_data_dir`, `--hdr_fps_samples`,
  `--hdr_invalid_intensity_ratio`, `--hdr_max_intensity`, `--skip_hdr`
* Recon output: `--recon_output_dir`
* Warping: `--warp_output_dir`, `--warp_depth_thresh_mm`,
  `--warp_smooth_depth1`, `--warp_smooth_ksize`, `--warp_smooth_sigma`,
  `--guided_r`, `--guided_eps`, `--guided_alpha`,
  `--nir_sharp_lo`, `--nir_sharp_hi`, `--swir_sharp_lo`, `--swir_sharp_hi`
* Pipeline: `--cam_types`, `--run_warp`

## How to run

From the repository root:

```bash
# Default scene = extra_scene, both cameras, then warp
python main.py

# Different scene + skip HDR if it has already been built
python main.py --scene_name my_scene --skip_hdr

# Only run NIR (no warp)
python main.py --cam_types nir --run_warp false
```
