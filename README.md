# Tensor Dynamic Mode Decomposition (Tensor DMD)

Research implementation and experimental evaluation of tensor-oriented Dynamic Mode Decomposition methods for spatiotemporal data.

**Project scope:** Five methods are implemented and compared: FFT-TDMD, Haar-TDMD, Cosine-TDMD, Unfolding DMD, and TT-DMD. Experiments include SST climate reconstruction, TT-DMD rank selection and validation, synthetic foreground-background benchmarks, and supplementary eigentube-norm analysis.

## Overview

Dynamic Mode Decomposition (DMD) extracts data-driven dynamical modes from sequential observations. Tensor DMD methods aim to exploit the multidimensional structure of spatiotemporal data instead of treating every snapshot only as a flattened vector.

This project implements and evaluates several tensor-oriented DMD variants and compares their reconstruction behavior, numerical error, and selected computational trade-offs.

## Implemented Methods

| Method | Approach |
|---|---|
| FFT-TDMD | Frequency-domain tensor DMD approach |
| Haar-TDMD | Haar wavelet-based approach |
| Cosine-TDMD | Cosine-transform-based approach |
| Unfolding DMD | DMD applied through matrix unfolding |
| TT-DMD | Tensor-train representation with rank truncation |

## Climate Reconstruction Experiment

The climate experiment uses Sea Surface Temperature (SST) data represented as a tensor of size 60 x 60 x 12.

| Method | MAE | RMSE |
|---|---:|---:|
| FFT-TDMD | 0.000999 | 0.001288 |
| Unfolding DMD | 0.000999 | 0.001288 |
| Haar-TDMD | 0.001106 | 0.001420 |
| Cosine-TDMD | 0.001340 | 0.001717 |
| TT-DMD | 0.040086 | 0.054975 |

## TT-DMD Rank Validation

| Configuration | Mean rolling validation error |
|---|---:|
| Full | 1.9674% |
| Medium | 2.0602% |
| Medium compact | 2.5611% |
| Low spatial | 3.0399% |
| Compact | 3.2956% |
| Small | ~9% |

## Synthetic Foreground-Background Benchmark

| Method | F1 Score | MAE |
|---|---:|---:|
| Temporal Median | 0.7725 | 0.005928 |
| FFT-TDMD | 0.7105 | 0.025780 |
| Haar-TDMD | 0.6848 | 0.040820 |
| Cosine-TDMD | 0.5887 | 0.042890 |

## Supplementary Eig​​entube-Norm Experiment

The eigentube-norm experiment explores frequency-domain tensor eigenvalue tracking and reconstruction. It is supplementary and is not claimed to be a complete implementation of Tensor Eigenvalue Decomposition (TEVD).

## Repository Structure

    Tensor_DMD_Project/
    |-- src/
    |-- results/
    |-- outputs/
    |-- README.md
    |-- requirements.txt

## Installation

Install dependencies:

    pip install -r requirements.txt

## Experiments

Experiment scripts are organized under src/. Climate comparison, rank validation, foreground-background experiments, and supplementary analysis can be run individually.

## Limitations

- Results depend on the datasets and parameter configurations used.
- TT-DMD performance is sensitive to tensor-train rank selection.
- Synthetic foreground-background benchmarks do not establish performance on all real-world video datasets.
- The eigentube-norm experiment is exploratory and is not a full TEVD reproduction.

## Reference

He, Hu, Lou, and Chen, Tensor Dynamic Mode Decomposition, IEEE Signal Processing Letters, 2026. DOI: 10.1109/LSP.2026.3673197.


