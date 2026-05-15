# BGMA-Net

<p align="center">
  <b>A Blur-Guided Multi-Attention Network Based on Left-Right Consistency for Gradual Defocus Deblurring in Binocular Images</b>
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.neucom.2025.131853"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.neucom.2025.131853-blue"></a>
  <a href="https://www.sciencedirect.com/science/article/pii/S0925231225025251"><img src="https://img.shields.io/badge/Neurocomputing-2026-9cf"></a>
  <a href="https://github.com/Zkichn/BGMA-Net"><img src="https://img.shields.io/badge/Code-PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <img src="https://img.shields.io/badge/Task-Stereo%20Defocus%20Deblurring-green">
</p>

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S0925231225025251">Paper</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#results">Results</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="assets/figures/teaser.svg" width="92%" alt="BGMA-Net teaser">
</p>

Official PyTorch implementation of **BGMA-Net**, a blur-guided multi-attention network for gradual defocus deblurring in binocular froth images. BGMA-Net uses blur-aware weighting to localize degraded regions, blur-guided multi-attention to strengthen feature restoration, and a left-right consistency framework to exploit stereo correspondence during training.

## News

- **2026.01**: BGMA-Net appears in *Neurocomputing*, Volume 661, Article 131853.
- **2025.10**: Paper accepted by *Neurocomputing*.
- **Code release**: Core model modules are available. Training scripts, checkpoints, and dataset preparation utilities are being organized.

## Highlights

- **Blur-aware weighting module (BAWM)** estimates local blur severity from froth-image structure and provides explicit guidance maps.
- **Blur-guided multi-attention (BGMA)** combines spatial, channel, and local multi-head attention to focus restoration on severely blurred regions.
- **Left-right consistency (LRC)** leverages stereo geometry during training without adding inference overhead.
- **Industrial validation** shows strong generalization on naturally defocused flotation-froth imagery.

## Method

<p align="center">
  <img src="assets/figures/overview.svg" width="92%" alt="Overall architecture of BGMA-Net with LRC">
</p>

BGMA-Net is built around three components:

1. **BAWM** generates blur weight maps by detecting blurred regions and estimating spatially varying blur severity.
2. **BGMA module** injects blur guidance into multi-scale feature learning and helps the network allocate more attention to degraded areas.
3. **LRC framework** reconstructs left/right views through disparity warping and adds a stereo-consistency loss during training.

<p align="center">
  <img src="assets/figures/bgma_module.svg" width="70%" alt="BGMA module">
</p>

<p align="center">
  <img src="assets/figures/lrc_framework.svg" width="78%" alt="LRC framework">
</p>

## Results

### Holopix50k

| Method | 25% Blur PSNR | 25% Blur SSIM | 50% Blur PSNR | 50% Blur SSIM | 75% Blur PSNR | 75% Blur SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DPANet | 36.945 | **0.9847** | 34.006 | 0.9544 | 31.532 | 0.9263 |
| BGMA-Net+ | 36.317 | 0.9731 | 33.214 | 0.9484 | 31.116 | 0.9212 |
| **BGMA-Net\*** | **37.136** | 0.9763 | **34.318** | **0.9587** | **31.770** | **0.9292** |

### Flotation Froth Dataset

| Method | 25% Blur PSNR | 25% Blur SSIM | 50% Blur PSNR | 50% Blur SSIM | 75% Blur PSNR | 75% Blur SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BaMBNet | 43.957 | 0.9934 | 35.129 | 0.9508 | 32.842 | 0.9361 |
| DPANet | 44.189 | **0.9942** | 35.319 | 0.9595 | 32.925 | 0.9353 |
| **BGMA-Net\*** | **44.322** | 0.9936 | **35.585** | **0.9625** | **33.227** | **0.9395** |

<p align="center">
  <img src="assets/figures/qualitative_results.svg" width="95%" alt="Qualitative comparison">
</p>

<p align="center">
  <img src="assets/figures/industrial_results.svg" width="92%" alt="Industrial defocus deblurring results">
</p>

## Repository Layout

```text
BGMA-Net/
├── BGMA-Net/
│   └── model/
│       ├── BAWM.py       # Blur-aware weighting module
│       ├── BGMA.py       # Blur-guided multi-attention module
│       ├── LRC.py        # Left-right consistency loss utilities
│       └── Network.py    # BGMA-Net backbone and stereo wrapper
├── assets/
│   └── figures/          # Paper figures for the project page
├── CITATION.cff
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Zkichn/BGMA-Net.git
cd BGMA-Net
conda create -n bgmanet python=3.10 -y
conda activate bgmanet
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if needed.

## Quick Start

The current release contains the core network modules. A minimal forward pass:

```python
import sys
import torch

sys.path.append("BGMA-Net")
from model.Network import BGMA_net_double

model = BGMA_net_double().eval()

left = torch.randn(1, 1, 248, 248)
right = torch.randn(1, 1, 248, 248)
left_weight = torch.randn(1, 1, 248, 248)
right_weight = torch.randn(1, 1, 248, 248)

with torch.no_grad():
    left_deblurred, right_deblurred = model(left, left_weight, right, right_weight)

print(left_deblurred.shape, right_deblurred.shape)
```

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{chen2026bgmanet,
  title = {A blur-guided multi-attention network based on left-right consistency for gradual defocus deblurring in binocular images},
  author = {Chen, Zekai and Tang, Zhaohui and Zhong, Yuze and Zhang, Hu and Dai, Zhien and Xie, Yongfang},
  journal = {Neurocomputing},
  volume = {661},
  pages = {131853},
  year = {2026},
  doi = {10.1016/j.neucom.2025.131853}
}
```

## Acknowledgements

This repository accompanies the paper published in *Neurocomputing*. The experiments use Holopix50k and an industrial flotation froth binocular-image dataset collected from a lead-zinc flotation plant.
