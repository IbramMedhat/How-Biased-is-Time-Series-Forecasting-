# Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased is Time Series Forecasting?

[![Conference](https://img.shields.io/badge/PAKDD-2026-blue.svg)](https://pakdd2026.org/)
[![Organization](https://img.shields.io/badge/Organization-ISMLL-red.svg)](https://www.ismll.uni-hildesheim.de/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased is Time Series Forecasting?"**, accepted at the 30th Pacific-Asia Conference on Knowledge Discovery and Data Mining (**PAKDD 2026**).

> **Note:** For the full theoretical analysis and additional experimental logs, please refer to our **[Extended Version on arXiv](https://arxiv.org/pdf/2502.09683)**.

## Summary
This work investigates experimental shortcomings in the Long-term Time Series Forecasting (LTSF) literature. We demonstrate that:
1. **Lookback Bias:** Fixed lookback windows (e.g., $L=96$) used in benchmarking often unfairly penalize specific architectures.
2. **Channel Dependency:** Standard benchmarks (Weather, ETT) often fail to distinguish between Channel-Independent (CI) and Channel-Dependent (CD) models due to low inter-channel causality.
3. **ODE Benchmarks:** We introduce a rigorous evaluation framework using chaotic ODE systems to better evaluate multivariate modeling.

---

## Setup & Installation

### 1. Environment
Create the conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate ltsf-bias
```

### 2. Datasets
All datasets, including the generated **ODE Suite**, can be downloaded from the following Google Drive folder:
**[Download Datasets](https://drive.google.com/drive/folders/1WTMTHwoyJVv8C0WtSh9Alc27xR37dnNI?usp=sharing)**

### 3. Implementation Details
Detailed hyperparameter (HP) ranges and the optimal parameters required to reproduce our results are documented in the **Appendix of the extended version [1]**.

---

## Usage

### 1. Running Experiments
Example scripts for the experiments are included in the `scripts/` directory. For example, to run the ETTh1 experiment:
```bash
sh scripts/run_ETTh1.sh
```

And an example for running granger causality analysis is 
```
sh scripts/run_granger_CellCycle.sh
```

### 2. Testing New Models
You can integrate new models under the `models/` directory. To evaluate them on the ODE suite:
1. Add your model architecture to `models/`.
2. Edit `run_longExp.py` to include the HP ranges and arguments expected for your model.

---

## Acknowledgments & Citations

### Codebase Attribution
This repository is built upon the **[LTSF-Linear (DLinear)](https://github.com/cure-lab/LTSF-Linear)** codebase. We thank the authors for their significant contribution to the LTSF benchmarking community.

### Citation
If you find this work or the ODE suite useful in your research, please cite (Official Conference version to be added soon):

```bibtex
@article{abdelmalak2025channel,
  title={Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased is Time Series Forecasting?},
  author={Abdelmalak, Ibram and Madhusudhanan, Kiran and Choi, Jungmin and Kloetergens, Christian and Yalavarit, Vijaya Krishna and Stubbemann, Maximilian and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:2502.09683},
  year={2025}
}
```
