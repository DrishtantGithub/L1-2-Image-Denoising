# L1/2 Regularization for Image Denoising

**CS754 Course Project — IIT Bombay**  
Yug Thosar (25M1101) · Drishtant Jain (24M1085)  
Guidance: Prof. Ajit Rajwade

---

## Overview

This project explores **L1/2 regularization** as a non-convex alternative to classical L1/L2 methods for sparse image denoising. We implement and compare three sparse coding approaches — OMP, L1 (ISTA), and L1/2 (IRLS) — and further extend IRLS into an **Unrolled Neural Network** architecture for faster inference.

The key finding: L1/2 achieves the best reconstruction quality (PSNR: 24.70 dB, RMSE: 0.058) while remaining significantly faster than L1 (ISTA).

---

## Repository Structure

```
.
├── OMP_ISTA/                  # Python: OMP and L1 (ISTA) baselines
│   ├── notebooks/
│   │   ├── 01_baseline_omp_ksvd.ipynb
│   │   └── 02_li_sparse_coding.ipynb
│   ├── utils/
│   └── data/
│       ├── cameraman.png
│       └── learned_dictionary_bsds500.mat
│
├── prev_trad_IRLS/            # MATLAB: Traditional IRLS implementation
│   ├── irls_l12_stable.m
│   ├── half_threshold.m
│   ├── ksvd_l12_train.m
│   ├── irls_validation.m
│   ├── phase2_run.m
│   ├── phase3_denoising.m
│   └── train_dict_phase2.m
│
├── Unrolled_IRLS/             # MATLAB: Unrolled IRLS-Net
│   ├── unrolled_NN_IRLS.m
│   ├── IRLSLayer.m
│   ├── IRLSLayer_tuning.m
│   ├── optimal_tuned_IRLS_unrolled_dict.m
│   ├── lambda_regulairze.m
│   └── tuning_visualise.m
│
├── compare_performance.m      # Top-level evaluation script
├── compare_performance_averaged.m
├── report.pdf
└── presentation.pdf
```

---

## Methods

| Method | Type | Language |
|---|---|---|
| OMP | Greedy baseline | Python |
| L1 / ISTA | Convex, soft-thresholding | Python |
| IRLS (L1/2) | Non-convex, iterative reweighted LS | MATLAB |
| Unrolled IRLS-Net | IRLS unrolled into fixed-depth network (K=5 layers) | MATLAB |

---

## Results

### OMP vs L1 vs L1/2

| Method | PSNR (dB) | RMSE | Time (s) |
|---|---|---|---|
| OMP | 20.81 | 0.091 | 2.59 |
| L1 (ISTA) | 23.75 | 0.065 | 121.87 |
| **L1/2 (IRLS)** | **24.70** | **0.058** | **18.95** |

### IRLS Variants

| Method | Mode | PSNR (dB) | RMSE | Time (s) |
|---|---|---|---|---|
| Patchwise IRLS | Distinct | 24.70 | 0.0582 | 18.95 |
| Patchwise IRLS | Sliding | 22.92 | 0.0715 | 70.88 |
| Unrolled-Net | Distinct | 22.65 | 0.0737 | 5.86 |
| Unrolled-Net | Sliding | 21.50 | 0.0842 | 22.48 |

The Unrolled-Net achieves a **~3x speedup** over traditional IRLS with only a modest drop in quality.

---

## Dataset

All experiments use the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) dataset. The pre-trained dictionary (`learned_dictionary_bsds500.mat`) is included in `OMP_ISTA/data/`.

---

## Requirements

**Python** (for OMP_ISTA notebooks):
- `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `jupyter`

**MATLAB** (for IRLS and Unrolled_IRLS):
- MATLAB R2021a or later recommended

---

## References

1. Xu et al. (2012). *L1/2 Regularization: A Thresholding Representation Theory and a Fast Solver.* IEEE TNNLS.
2. Ba et al. *Convergence and Stability of Iteratively Re-weighted Least Squares Algorithms for Sparse Signal Recovery.* IEEE TSP.
