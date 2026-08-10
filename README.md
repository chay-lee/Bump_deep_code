# 3D coordinate-aware deep learning for estimating heights of micro-bumps from projections

by Chaeyoung Lee, Seungmi Oh, Yujin Jang, and Jeongtae Kim

---

## Abstract

The precise inspection of micro-bump height is critical for ensuring the reliability of high-density semiconductor packaging. Existing metrology solutions often face a trade-off between inspection speed and measurement accuracy, and conventional methods struggle to suppress optical noise without blurring structural boundaries and fail to correct position-dependent errors. To address these limitations, we propose an end-to-end deep learning framework that reconstructs complete 3D bump surfaces directly from raw triangulation data. By estimating probabilistic height distributions from 3D CNN features, our architecture robustly recovers structural details while suppressing optical noise. We further integrate Coordinate Convolution to resolve position-dependent spatial distortions. Additionally, to build an end-to-end framework, we incorporate a differentiable soft-argmax block and a 2D CNN refiner, along with a joint loss formulation that optimizes structural fidelity and estimation stability. Experimental results demonstrate that the proposed framework achieves both sub-micrometer accuracy and robust scan-to-scan repeatability.

---

## Getting Started

<details>
<summary><b>Dataset Download</b></summary>

The **Coin Bump Dataset** used in this study is available on Zenodo.

**DOI:** https://doi.org/10.5281/zenodo.21834980

The dataset includes optical triangulation measurements and corresponding CSI ground-truth height maps of semiconductor coin bumps. 
```bash
unzip Coin_bump_dataset.zip
```
</details>

<details>
<summary><b>Environment Settings</b></summary>

- **OS**: Ubuntu 20.04.1 LTS
- **Language**: Python 3.8.10
- **Dependencies**: Listed in `requirements.txt`

```bash
# Clone the repository
git clone [https://github.com/chay-lee/Bump_deep_code.git](https://github.com/chay-lee/Bump_deep_code.git)
mv Bump_deep_code codes
cd codes

# Create and activate a virtual environment
# Option A: using virtualenv
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U -r requirements.txt

# Option B: using pipenv
pipenv install Pipfile
pipenv shell
```

</details>

<details>
<summary><b>Train and Evaluate</b></summary>

To facilitate reproduction, we provide a unified shell script (`run.sh`) pre-configured with the hyperparameters detailed in the paper. You can execute the training and evaluation pipelines using the following commands.

**Usage:**

```bash
bash run.sh [COMMAND] [VARIANT]
```

**[COMMAND] Options:**

train: Train the selected model from scratch.

test: Evaluate the trained model (requires a saved checkpoint from the train step).

all: Execute training followed immediately by evaluation.

**[VARIANT] Options:**

proposed: The final proposed framework (Recon + Consis + TV).

recon_only: The baseline model performing pure 3D reconstruction.

recon_consis: Reconstruction with consistency loss applied.

wo_coordconv: An ablation model (the proposed framework without CoordConv).

**Examples:**

```bash
# 1. Train the final proposed model
bash run.sh train proposed

# 2. Evaluate the proposed model (ensure you have trained the model first)
bash run.sh test proposed

# 3. Train and evaluate the baseline model sequentially in one go
bash run.sh all recon_only
```

</details>

---

## Network Architecture

The proposed deep learning-based bump height estimator processes the 3D probability volume constructed from raw frame data through a multi-stage pipeline designed to mitigate optical noise and position-dependent errors. Specifically, a 3D CNN conditioned on explicit spatial coordinates from a CoordConv block estimates refined probabilistic features. A soft-argmax block then projects these estimated features into an initial 2D height map, which a final 2D CNN subsequently refines to encourage 2D surface continuity.

![Overall Architecture](assets/overall_architecture_v4.png)

---
## Results

Our framework is evaluated on both **pixel-wise reconstruction accuracy** ($E_p$) and **scan-to-scan metrological repeatability** ($\sigma_p, \sigma_b$).

### 1. Quantitative Results

> **Note:** All metrics are expressed in $10^{-2}~\mu\text{m}$. Bold text indicates optimal performance. Values are presented as mean ± standard deviation across three distinct random seeds.

| Method | $E_p$ (Pixel MAE) | $\sigma_p$ (Pixel STD) | $E_b$ (Bump Bias) | $\sigma_b$ (Bump STD) |
| :--- | :---: | :---: | :---: | :---: |
| **MAP** | 164.67 | 7.72 | 6.64 | 5.75 |
| **Recon Only** | **17.89** ± 0.28 | 8.04 ± 0.22 | 6.50 ± 0.95 | 4.79 ± 0.21 |
| **Recon + Consis** | 20.94 ± 0.57 | 7.73 ± 0.27 | 7.75 ± 0.31 | 4.35 ± 0.08 |
| **Proposed** | 20.29 ± 0.54 | **6.69** ± 0.07 | **5.25** ± 0.76 | **4.28** ± 0.05 |
| **w/o CoordConv** | 37.02 ± 0.12 | 9.95 ± 0.37 | 25.46 ± 1.07 | 5.14 ± 0.05 |

### 2. Qualitative Results

<div align="center">
  <table style="border:none;">
    <tr>
      <td align="center"><img src="assets/recon/Full_3d_GT.png" alt="GT" width="100%" /><br><b>(a) GT</b></td>
      <td align="center"><img src="assets/recon/Full_3d_MAP.png" alt="MAP" width="100%" /><br><b>(b) MAP</b></td>
      <td align="center"><img src="assets/recon/Full_3d_Pred.png" alt="Proposed" width="100%" /><br><b>(c) Proposed</b></td>
    </tr>
  </table>
  <p>
    <em>Qualitative comparison of the global 3D surface reconstructions. The panels illustrate (a) the ground truth (GT), (b) the MAP baseline, and (c) the Proposed model. The horizontal axes (X, Y) and vertical axis (H) represent spatial position and height in micrometers (μm).</em>
  </p>
</div>

---

## Credits

The coin bump test wafer used for data acquisition was provided by ATI.

We established the Maximum a Posteriori (MAP) estimation-based metrology pipeline as our primary statistical baseline for comparative analysis.

This repository contains our independent implementation of the joint 3D CNN, CoordConv layers, and Soft-argmax projection blocks tailored for wafer-level micro-bump metrology.

---

## License

This project is licensed under the [MIT License](LICENSE)
