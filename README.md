# 3D coordinate-aware deep learning for estimating heights of micro-bumps from projections

Chaeyoung Lee, Seungmi Oh, Yujin Jang, and Jeongtae Kim

## Abstract

In advanced semiconductor packaging, micro-bumps serve as electrical and thermal interconnects between stacked dies, making precise height inspection critical for structural and functional reliability. High-precision metrology methods can provide accurate height measurements but often face limitations in inspection speed, whereas optical triangulation enables rapid inspection but is susceptible to optical noise and position-dependent measurement errors. To alleviate the problems of optical triangulation-based methods, a previous study applied maximum a posteriori (MAP) estimation with a quadratic smoothness constraint, but yielded only limited improvement as blurred bump boundaries and position-dependent errors remain. To address these limitations, we propose an end-to-end convolutional neural network (CNN)-based deep learning system that utilizes a 3D CNN to refine the probability distribution of heights with coordinate information for suppressing optical noise and compensating for the position-dependent error, followed by a differentiable soft-argmax block and a 2D CNN for accurate height estimates. In experiments using real coin-bump data, the proposed system yielded more accurate height maps and bump peak height estimates as well as improved scan-to-scan repeatability.

## Installation

Use Python 3.10 or newer. GPU execution requires a compatible NVIDIA driver and PyTorch installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

The Coin Bump Dataset is available on [Zenodo](https://doi.org/10.5281/zenodo.23488779).

Download and extract the dataset, then set `DATASET_PATH` in `run.sh` to the directory containing `train`, `val`, and `repeat_test`.

## Training and evaluation

Configure paths, model settings, GPU selection, and worker counts at the top of `run.sh`, then run:

```bash
bash run.sh train   # Train
bash run.sh test    # Evaluate
bash run.sh all     # Train, then evaluate
```

- `MODEL_ROOT`: directory for model checkpoints and results.
- `MODEL_FEATURE`: leave empty for an automatic name, or enter an existing model folder name for testing.
- `MODEL_MODE`: `final`, `mini`, or `both`.
- `NUM_WORKERS`: DataLoader workers; set to `0` for loading in the main process.
- `EVAL_WORKERS`: CPU workers for bump-height evaluation; set to `1` for serial evaluation.

Relative paths are resolved from the project directory. `EXPERIMENT_NAME` is a naming label; configure loss weights and architecture flags explicitly. Use a distinct model name when changing settings not included in the automatic name.

## Network architecture

The network combines coordinate-aware 3D convolution, a differentiable soft-argmax block, and a 2D CNN for height-map refinement.

![Network architecture](assets/overall_architecture_v4.png)

## Evaluation metrics

| Metric | Description |
| --- | --- |
| $E_p$ | Pixel-wise mean absolute error |
| $\sigma_p$ | Pixel-wise repeatability across repeated scans |
| $\sigma_s$ | Pixel-wise repeatability in substrate regions |
| $E_b$ | Bump peak-height mean absolute error |
| $\sigma_b$ | Bump peak-height repeatability across repeated scans |

All metrics are expressed in $10^{-2}\,\mu\mathrm{m}$. Lower values indicate better performance.

## Results

The following results are reported in the paper. Learned-model results are presented as mean ± standard deviation across three random seeds. Bold values indicate the best performance.

| Method | $E_p$ | $\sigma_p$ | $\sigma_s$ | $E_b$ | $\sigma_b$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAP | 164.74 | 7.47 | 10.92 | 13.59 | 4.20 |
| Proposed ($L_r$) | **28.53 ± 0.32** | 7.67 ± 0.07 | 4.78 ± 0.65 | 12.67 ± 0.35 | 3.58 ± 0.16 |
| Proposed ($L_r + L_c$) | 32.61 ± 0.82 | 6.15 ± 0.54 | 3.73 ± 0.15 | 13.34 ± 0.53 | 2.56 ± 0.33 |
| Proposed ($L_r + L_c + L_t$) | 29.38 ± 0.45 | **4.12 ± 0.67** | **1.99 ± 0.75** | **11.19 ± 0.56** | **1.79 ± 0.08** |
| Proposed (w/o CoordConv) | 43.36 ± 1.10 | 9.13 ± 0.42 | 3.83 ± 0.11 | 13.04 ± 0.92 | 3.03 ± 0.05 |
| Proposed (w/o 2D CNN) | 34.38 ± 0.07 | 5.33 ± 0.24 | 2.79 ± 0.57 | 13.76 ± 0.88 | 3.67 ± 0.19 |

## Credits and license

The coin-bump test wafer used for data acquisition was provided by ATI.

This project is licensed under the [MIT License](LICENSE).