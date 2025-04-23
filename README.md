# tadgan_sap
This repository contains a complete pipeline for time series anomaly detection on SAP financial transaction data using a modified TadGAN architecture. The project is developed as part of a Master's thesis in collaboration with Schwarz Group and aims to support audit teams with feature-level, interpretable anomaly detection directly integrated into existing dashboards.

## Project Objectives

- Detect hidden anomalies in transactional data (e.g., vendor misuse, off-cycle postings).

- Reduce false positives and enhance explainability for auditors.

- Leverage sequence modeling to respect temporal dependencies in data.

- Enable dashboard-ready outputs for integration into SAP-based systems.

## Repository Structure
sumitrra_sap/
├── preprocessing/
│   ├── tadgan_preprocessing.ipynb       # Full data cleaning, encoding & scaling
│   ├── sequence_generator.ipynb         # Convert tabular data into (N, seq_len, D) tensors
│
├── model/
│   ├── tadgan_v17.py                    # Generator & Discriminator training script with AMP
│
├── analysis/
│   ├── anomaly_analysis.ipynb           # Feature- and time-based anomaly detection
│   ├── visualization_tools.ipynb        # Heatmaps, scatter plots, sequence mapping
│
├── data/                                # (Local or cloud-stored data not in GitHub)
│
└── README.md                            # You're here!

## Preprocessing Overview

- Date parsing (budat, cpudt, aedat, etc.)

- Cyclical encoding of month/day

- Missing value treatment:

 - Binary → Fill with 0

 - Numeric → Median

 - Categorical → Mode

- Encoding:
  
  - High-cardinality: frequency encoding
  
  - Low-cardinality: one-hot encoding

- MinMax scaling of all non-binary features

- Final dataset: Fully numeric and normalized

## Model: TadGAN Architecture

- Generator: LSTM Encoder-Decoder

- Discriminator: LSTM + Dense output

- Loss functions:

  - BCEWithLogits (adversarial)
  
  - MSE (reconstruction)

- Training:
  
  - Mixed precision (AMP) for speed
  
  - Reconstruction error tracked across epochs
  
  - Threshold computed using 99th percentile

## Post-Hoc Analysis & Explainability

- Hybrid anomaly detection:

  - Time-based threshold (rolling window mean + k·std)
  
  - Feature-based threshold (MAD + percentile)

- Explainable outputs:

  - Heatmaps of anomalous features per sequence
  
  - Scatter plots with feature names
  
  - Sequence-to-original-row mapping

## Key Features

- Temporal modeling using overlapping sequences

- Time + feature anomaly explainability

- Dynamic thresholding via rolling stats

- High audit-readiness with interpretable feature triggers

## Requirements 
- Python ≥ 3.8

- PyTorch ≥ 1.10

- NumPy, Pandas, Matplotlib, Seaborn

- Optional: GPU for faster GAN training

