# End-to-End Deep Learning Framework for 3D Morphological Reconstruction and Height Estimation of Micro-Bumps

by Chaeyoung Lee, Seungmi Oh, Yujin Jang, and Jeongtae Kim

---

## Abstract

The precise inspection of micro-bump height is critical for ensuring the reliability of high-density semiconductor packaging. Existing metrology solutions often face a trade-off between inspection speed and measurement accuracy, and conventional methods struggle to suppress optical noise without blurring structural boundaries and fail to correct position-dependent errors. To address these limitations, we propose an end-to-end deep learning framework that reconstructs complete 3D bump surfaces directly from raw triangulation data. By estimating probabilistic height distributions from 3D CNN features, our architecture robustly recovers structural details while suppressing optical noise. We further integrate Coordinate Convolution to resolve position-dependent spatial distortions. Additionally, to build an end-to-end framework, we incorporate a differentiable soft-argmax block and a 2D CNN refiner, along with a joint loss formulation that optimizes structural fidelity and estimation stability. Experimental results demonstrate that the proposed framework achieves both sub-micrometer accuracy and robust scan-to-scan repeatability.

---

## Getting Started

<details>
<summary><b>Dataset Download</b></summary>

*(The dataset download link and instructions will be updated upon paper publication.)*

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

We provide a unified shell script (`run.sh`) that automatically loads the exact hyperparameter configurations reported in the paper (Table II). You can seamlessly train and evaluate the models with a single command.

**Usage:**

```bash
bash run.sh [COMMAND] [VARIANT]
```

**[COMMAND] Options:**

train: Train the selected model from scratch.

test: Evaluate a pre-trained model (calculates MAE, Bias, and repeatability metrics).

all: Execute full training followed immediately by evaluation.

**[VARIANT] Options:**

proposed: The final proposed framework (Recon + Consis + TV).

recon_only: The baseline model performing pure 3D reconstruction.

recon_consis: Reconstruction with consistency loss applied.

wo_coordconv: An ablation model (the proposed framework without CoordConv).

**Examples:**

```bash
# 1. Train the final proposed model
bash run.sh train proposed

# 2. Evaluate the pre-trained proposed model (requires weights in the /weights directory)
bash run.sh test proposed

# 3. Train and evaluate the baseline model sequentially
bash run.sh all recon_only
```

</details>

---

## Network Architecture

The proposed deep learning-based bump height estimator processes the 3D probability volume constructed from raw frame data through a multi-stage pipeline designed to mitigate optical noise and position-dependent errors. Specifically, a 3D CNN conditioned on explicit spatial coordinates from a CoordConv block estimates refined probabilistic features. A soft-argmax block then projects these estimated features into an initial 2D height map, which a final 2D CNN subsequently refines to encourage 2D surface continuity.

![Overall Architecture](assets/overall_architecture_v4.png)

---

## Results

## Results

<details>
<summary><b>Main Results</b></summary>

> **Note:** All metrics are expressed in $10^{-2}~\mu\text{m}$. Bold text indicates optimal performance, and values are presented as mean ± standard deviation across three distinct random seeds.

| Method | $E_p$ | $\sigma_p$ | $E_b$ | $\sigma_b$ |
| :--- | :---: | :---: | :---: | :---: |
| **MAP** | 164.67 | 7.72 | 6.64 | 5.75 |
| **Recon Only** | **17.89** ± 0.28 | 8.04 ± 0.22 | 6.50 ± 0.95 | 4.79 ± 0.21 |
| **Recon + Consis** | 20.94 ± 0.57 | 7.73 ± 0.27 | 7.75 ± 0.31 | 4.35 ± 0.08 |
| **Proposed (Recon + Consis + TV)** | 20.29 ± 0.54 | **6.69** ± 0.07 | **5.25** ± 0.76 | **4.28** ± 0.05 |
| **w/o CoordConv (Ablation)** | 37.02 ± 0.12 | 9.95 ± 0.37 | 25.46 ± 1.07 | 5.14 ± 0.05 |

- *$E_p$: Pixel-wise Mean Absolute Error (MAE) / $\sigma_p$: Pixel-wise Standard Deviation*
- *$E_b$: Bump-level Bias (Error) / $\sigma_b$: Bump-level Standard Deviation*
- *The Proposed framework achieves the optimal balance between topographic fidelity and metrological repeatability.*

</details>

---

## Credits

We established the Maximum a Posteriori (MAP) estimation-based metrology pipeline as our primary statistical baseline for comparative analysis.

This repository contains our independent implementation of the joint 3D CNN, CoordConv layers, and Soft-argmax projection blocks tailored for wafer-level micro-bump metrology.

---

## License

This project is licensed under the [MIT License](LICENSE)
