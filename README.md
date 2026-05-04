# VNIR — `clean_code`

A reorganized version of the VNIR pipeline that replaces the previous
mix of scripts and Jupyter notebooks with a single end-to-end driver
(`clean_code/main.py`).

The old workflow was:

```
hdr.ipynb            -> dumps  dataset/hdr_dataset/<cam>_hdr_<scene>.npy
main.py              -> stereo images -> depth -> hyperspectral recon
warpped/warp_gif.py  -> warp SWIR results into the NIR view
```

The new workflow runs everything from one command:

```
python -m clean_code.main --scene_name <scene>
```

## Pipeline overview

`clean_code/main.py` runs five steps in order. Steps 2–4 are run
once per camera (`nir`, `swir`); step 1 happens for both cameras
together; step 5 is run once at the end.

| # | Step | Module | What it does |
|---|------|--------|--------------|
| 1 | Stereo image preparation | `vnir_utils/utils.py::make_stereo_images` | Max-projects galvo-scanned captures into a single sharp image per camera and writes `<scene>_depth/<cam>/capture_0000.png`. |
| 2 | HDR generation **[NEW]** | `vnir_utils/hdr.py::make_hdr_npy` | For each angle index, fuses the multi-exposure captures (under `<hdr_data_dir>/<scene>_<fps>fps/<cam>`) using a trapezoid weight + black-frame subtraction and writes `<hdr_data_dir>/<cam>_hdr_<scene>.npy`. Replaces `hdr.ipynb`. |
| 3 | Depth reconstruction | `vnir_recon/recon_depth.py` | Rectifies the stereo pair, runs Foundation Stereo, then inverse-rectifies the disparity into the original camera frame. Caches `<scene>_depth/<cam>_depth.npy`. |
| 4 | Hyperspectral recon | `vnir_recon/recon_vnir.py` | Adam-optimizes per-pixel hyperspectral reflectance against the HDR stack (in-memory — no disk round-trip). Saves `<recon_output_dir>/<cam>_<scene>.npy`. |
| 5 | Warping **[NEW integration]** | `vnir_utils/warp.py` | Detail-transfers using a guided filter on the SWIR side, then warps each SWIR wavelength into the NIR camera view using both depth maps + an occlusion gate. PNGs go to `<warp_output_dir>/<scene>/`. |

## Key change vs. the old code

`vnir_recon/recon_vnir_crop.py` previously did:

```python
GT_IMAGE = np.load('./dataset/hdr_dataset/%s_hdr_%s.npy' % (cam_type, args.date[0])) ...
```

i.e. it required the user to have run `hdr.ipynb` ahead of time.
`clean_code/vnir_recon/recon_vnir.py` instead exposes a new signature:

```python
ReconVNIR(args, cam_type).optimization(L_lambda, wvls, depths, hdr_stack)
```

The HDR stack is built fresh in step 2 by `make_hdr_npy`, kept in
RAM, and handed straight to the optimizer in step 4. It is also
written to `<hdr_data_dir>/<cam>_hdr_<scene>.npy` so reruns can
short-circuit by passing `--skip_hdr`.

## Project layout

```
clean_code/
├── main.py
├── README.md
├── vnir_utils/
│   ├── argparser.py        ← original args + [NEW] HDR / warp / pipeline args
│   ├── hdr.py              ← [NEW] replaces hdr.ipynb
│   ├── warp.py             ← [NEW] library port of warp_gif.py
│   ├── utils.py
│   ├── rectify.py
│   ├── inverse_rectify.py
│   └── depth_utils.py
├── vnir_sl/
│   ├── render.py           ← simulated rendering (gaussian_render_crop)
│   └── datatools.py
└── vnir_recon/
    ├── recon_depth.py      ← Foundation Stereo wrapper
    └── recon_vnir.py       ← in-memory HDR variant
```

External dependencies (must remain at the repo root and were not
modified):

* `FoundationStereo/` — pretrained stereo model
* `guided_filter/` — guided-filter implementation used by warping
* `calibration/` — radiometric, prism, and stereo calibration data
* `dataset/` — captured frames, radiometric data, and HDR raw captures

## Argparser additions

All new arguments live in `clean_code/vnir_utils/argparser.py`. They
are introduced under banners that read

```python
# =====================================================================
# ### NEW (clean_code) ###
# =====================================================================
```

Each individual argument that was added is also tagged with `[NEW]`
in its `help` string. They cover:

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
python -m clean_code.main

# Different scene + skip HDR if it has already been built
python -m clean_code.main --scene_name my_scene --skip_hdr

# Only run NIR (no warp)
python -m clean_code.main --cam_types nir --run_warp false
```

## Files / folders intentionally not used

The user request explicitly excluded these from the refactor:

* All visualization notebooks (`*.ipynb`), generated `.png`/`.svg`/`.gif`,
  cached `.npy` files at the repo root
* `rebuttal_plan/`, `revision/`, `supple_*`, `blur_test/`,
  `rendered_image_results/`, `wandb/` — debugging / paper artifacts
* `old_codes/` — superseded by this refactor

The original `main.py`, `vnir_recon/`, `vnir_utils/`, `vnir_sl/`,
and `warpped/warp_gif.py` are left untouched in case you need to
diff or roll back.
