# Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased is Time Series Forecasting

[![Conference](https://img.shields.io/badge/PAKDD-2026-blue.svg)](https://pakdd2026.org/)
[![Organization](https://img.shields.io/badge/Organization-ISMLL-red.svg)](https://www.ismll.uni-hildesheim.de/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased is Time Series Forecasting?"**, accepted at the 30th Pacific-Asia Conference on Knowledge Discovery and Data Mining (**PAKDD 2026**).

## Summary
This work investigates experimental shortcomings in the Long-term Time Series Forecasting (LTSF) literature. We demonstrate that:
1. **Lookback Bias:** Fixed lookback windows (e.g., $L=96$) used in benchmarking often unfairly penalize specific architectures.
2. **Channel Dependency:** Standard benchmarks (Weather, ETT) often fail to distinguish between Channel-Independent (CI) and Channel-Dependent (CD) models due to low inter-channel causality.
3. **ODE Benchmarks:** We introduce a rigorous evaluation framework using chaotic ODE systems to better evaluate multivariate modeling.

---

## Setup & Installation

### 1. Environment
Create the conda environment using the provided `ltsf-bias.yml` file:
```bash
conda env create -f environment.yml
conda activate ltsf-bias
```

### 2. Datasets

You can download all the datasets from the following google drive including the generated ODE suit : ![Google Drive](https://drive.google.com/drive/folders/1WTMTHwoyJVv8C0WtSh9Alc27xR37dnNI?usp=sharing)

### 3. Running Scripts

Example scripts for some of the experiments are included in the scripts/ directory



