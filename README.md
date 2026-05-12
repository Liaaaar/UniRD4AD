# UniRD4AD

PyTorch implementation for multi-class unsupervised industrial anomaly detection based on reverse distillation with latent-space regularization.

This repository follows the paper:

**Enhancing reverse distillation model for multi-class unsupervised anomaly detection via latent space regularization**  
Kaiwen Fu, Fei Qi, Xiaotian Wang, Kun Liu  
*Neurocomputing*, Volume 660, Article 131819, 2026  
DOI: [10.1016/j.neucom.2025.131819](https://doi.org/10.1016/j.neucom.2025.131819)

## Overview

UniRD4AD trains a single anomaly detection model across multiple object categories. It uses a frozen ImageNet-pretrained ResNet encoder as the teacher, a feature fusion bottleneck module, and a reverse decoder as the student. Normal training images are used only; anomalies are detected by the reconstruction discrepancy between teacher and student feature maps.

Key components in this implementation:

- Multi-class training for MVTec AD, VisA, and Real-IAD.
- Reverse distillation with ResNet / Wide-ResNet backbones.
- Latent-space cluster regularization on fused embeddings.
- Image-level AUROC, pixel-level AUROC, and pixel-level AUPRO evaluation.
- Training resume, best-checkpoint saving, training curves, and CSV result export.

## Repository Structure

```text
.
├── train.py                    # Training entry point
├── infer.py                    # Evaluation / inference entry point
├── train.sh                    # Example shell script
├── datasets/
│   └── datasets.py             # MVTec AD, VisA, Real-IAD loaders
├── models/
│   ├── encoders/resnet.py      # Teacher encoder and bottleneck fusion layer
│   ├── decoders/de_resnet.py   # Reverse ResNet decoder
│   └── losses/losses.py        # RD loss and cluster loss
├── utils/
│   └── utils.py                # Metrics and anomaly-map utilities
├── checkpoints/                # Example checkpoints and logs
└── logs/                       # Training logs and result files
```

## Installation

Create a Python environment and install the dependencies:

```bash
conda create -n unird4ad python=3.9 -y
conda activate unird4ad

# Install PyTorch according to your CUDA version:
# https://pytorch.org/get-started/locally/

pip install torchvision numpy pandas scipy scikit-image scikit-learn matplotlib tqdm pillow
```

If you use `--cluster_loss ssot`, also install:

```bash
pip install geomloss
```

The encoder loads ImageNet weights through `torch.hub`, so the first run may download pretrained ResNet weights.

## Datasets

### MVTec AD

Set `--data_root` to the directory that contains all MVTec classes:

```text
mvtec/
├── bottle/
│   ├── train/good/*.png
│   ├── test/good/*.png
│   ├── test/<defect_type>/*.png
│   └── ground_truth/<defect_type>/*.png
├── cable/
└── ...
```

### VisA

The loader expects the official VisA split file:

```text
visa/
├── split_csv/1cls.csv
├── candle/
├── capsules/
└── ...
```

### Real-IAD

The loader expects Real-IAD JSON splits and the `realiad_1024` image directory:

```text
Real-IAD/
├── realiad_jsons/realiad_jsons/*.json
└── realiad_1024/
    ├── audiojack/
    ├── bottle_cap/
    └── ...
```

## Training

Train on MVTec AD:

```bash
python train.py \
  --dataset mvtec \
  --data_root /path/to/mvtec \
  --img_size 256 \
  --backbone wide_resnet50_2 \
  --rd_loss cosine \
  --cluster_loss cosine \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.01 \
  --save_path checkpoints/mvtec
```

Train on VisA:

```bash
python train.py \
  --dataset visa \
  --data_root /path/to/visa \
  --img_size 256 \
  --backbone wide_resnet50_2 \
  --rd_loss cosine \
  --cluster_loss cosine \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.01 \
  --save_path checkpoints/visa
```

Resume training:

```bash
python train.py \
  --dataset mvtec \
  --data_root /path/to/mvtec \
  --resume checkpoints/mvtec/best_model.pth \
  --save_path checkpoints/mvtec
```

Supported options:

- `--dataset`: `mvtec`, `visa`, `realiad`
- `--backbone`: `resnet18`, `resnet34`, `resnet50`, `wide_resnet50_2`, `wide_resnet101_2`
- `--rd_loss`: `cosine`, `arc`, `ssim`, `mse`, `mae`
- `--cluster_loss`: `cosine`, `arc`, `ssim`, `mse`, `mae`, `ssot`

Training saves:

- `best_model.pth`: best checkpoint selected by mean image AUROC, pixel AUROC, and pixel AUPRO
- `best_results.csv`: per-class and mean metrics
- `monitor_traning.png`: training and evaluation curves

## Inference

Evaluate a trained checkpoint:

```bash
python infer.py \
  --dataset mvtec \
  --data_root /path/to/mvtec \
  --checkpoint checkpoints/mvtec/best_model.pth \
  --backbone wide_resnet50_2 \
  --img_size 256
```

If you use the example checkpoint file in this repository:

```bash
python infer.py \
  --dataset mvtec \
  --data_root /path/to/mvtec \
  --checkpoint checkpoints/mvtec_best_model.pth
```

The script prints image-level AUROC, pixel-level AUROC, and pixel-level AUPRO for every class and their mean values.

## Results

Current reproduced results from the included logs/checkpoints:

| Dataset | Image AUROC | Pixel AUROC | Pixel AUPRO |
| --- | ---: | ---: | ---: |
| MVTec AD | 0.986 | 0.980 | 0.943 |
| VisA | 0.966 | 0.984 | 0.932 |

Results can vary with hardware, random seed, backbone, image size, dataset version, and PyTorch/CUDA versions.

## Citation

If this repository or the paper is useful for your work, please cite:

```bibtex
@article{fu2026enhancing,
  title   = {Enhancing reverse distillation model for multi-class unsupervised anomaly detection via latent space regularization},
  author  = {Fu, Kaiwen and Qi, Fei and Wang, Xiaotian and Liu, Kun},
  journal = {Neurocomputing},
  volume  = {660},
  pages   = {131819},
  year    = {2026},
  doi     = {10.1016/j.neucom.2025.131819}
}
```

## Acknowledgements

This project builds on reverse distillation for anomaly detection and uses ResNet-style encoders/decoders with ImageNet-pretrained teacher features.
