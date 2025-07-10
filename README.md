# Mangrove3D Point Cloud Segmentation

PyTorch implementation of PointNet++ for Mangrove3D point cloud semantic segmentation. Classifies points into 5 ecological classes: Ground, Stem, Canopy, Roots, and Objects.

## Quick Start

### 1. Setup Environment
```bash
conda create -n mangrove3d python=3.8 numpy tqdm pandas pyyaml -y
conda activate mangrove3d
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -y
```

### 2. Prepare Data
Organize your data:
```
data/
└── train_val/
    ├── area1_pcd_1_color.csv
    ├── area1_pcd_1_refined.label
    └── ...
```

### 3. Configure
Edit `params/config.yaml`:
```yaml
data:
  root_dir: "/path/to/your/data"
  val_ratio: 0.25
```

### 4. Train & Test
```bash
# Train with defaults
python train_semseg_mangrove3d.py

# Quick training
python train_semseg_mangrove3d.py --config params/quick_config.yaml

# High-quality training
python train_semseg_mangrove3d.py --config params/high_quality_config.yaml

# Test
python test_semseg_mangrove3d.py
```

## Configuration Files

- `params/config.yaml` - Default balanced settings
- `params/quick_config.yaml` - Fast training for development
- `params/high_quality_config.yaml` - Best results

Override any parameter via command line:
```bash
python train_semseg_mangrove3d.py --batch_size 16 --learning_rate 0.002
```

## Troubleshooting

**Out of Memory**: Reduce batch size or points per sample
```bash
python train_semseg_mangrove3d.py --batch_size 8 --npoint 2048
```

**Data Loading Issues**: Check file naming (`*_color.csv`, `*_refined.label`) and paths in config

**Poor Performance**: Use high-quality config or increase epochs
```bash
python train_semseg_mangrove3d.py --config params/high_quality_config.yaml
```
