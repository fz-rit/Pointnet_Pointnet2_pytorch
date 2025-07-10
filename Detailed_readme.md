# Mangrove3D Point Cloud Segmentation

A PyTorch implementation of PointNet++ optimized for Mangrove3D point cloud semantic segmentation. This repository provides a complete framework for training and testing point cloud segmentation models on mangrove ecosystem data.

## Overview

This project uses PointNet++ deep learning architecture to perform semantic segmentation on 3D point clouds of mangrove ecosystems. The model classifies each point into one of five ecological classes: Ground, Stem, Canopy, Roots, and Objects.

### Key Features

- **Advanced Data Loading**: Smart data splitting with configurable train/validation ratios
- **YAML Configuration**: Centralized parameter management for easy experimentation
- **Flexible Training**: Support for various optimization strategies and augmentation techniques
- **Comprehensive Evaluation**: Multiple metrics including IoU, accuracy, and class-specific performance
- **Visualization**: Built-in tools for result visualization and analysis
- **Reproducible Research**: Deterministic splits and comprehensive parameter tracking

## Quick Start

### 1. Environment Setup
```bash
conda create -n mangrove3d python=3.8 numpy tqdm pandas pyyaml -y
conda activate mangrove3d
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -y
```

### 2. Data Preparation
Organize your Mangrove3D data in one of these structures:

**Option A: Single folder (Recommended)**
```
data/
└── train_val/
    ├── area1_pcd_1_color.csv
    ├── area1_pcd_1_refined.label
    ├── area2_pcd_2_color.csv
    ├── area2_pcd_2_refined.label
    └── ...
```

**Option B: Separate folders**
```
data/
├── train/
│   ├── area1_pcd_1_color.csv
│   ├── area1_pcd_1_refined.label
│   └── ...
├── val/
│   ├── area2_pcd_2_color.csv
│   ├── area2_pcd_2_refined.label
│   └── ...
└── test/
    ├── area3_pcd_3_color.csv
    ├── area3_pcd_3_refined.label
    └── ...
```

### 3. Basic Configuration
Set your data path in `params/config.yaml`:

```yaml
data:
  root_dir: "/path/to/your/mangrove3d/data"
  val_ratio: 0.25  # 25% for validation
  random_seed: 42  # For reproducible splits
```

### 4. Training
```bash
# Quick start with defaults
python train_semseg_mangrove3d.py

# Override specific parameters
python train_semseg_mangrove3d.py --epoch 100 --batch_size 16

# Use pre-configured setups
python train_semseg_mangrove3d.py --config params/quick_config.yaml        # Fast training
python train_semseg_mangrove3d.py --config params/high_quality_config.yaml # Best results
```

### 5. Testing
```bash
# Basic testing
python test_semseg_mangrove3d.py

# High-accuracy testing with more votes
python test_semseg_mangrove3d.py --num_votes 10

# Test specific model
python test_semseg_mangrove3d.py --model_path ./checkpoints/best_model.pth
```

## Configuration System

The project uses a comprehensive YAML-based configuration system for easy parameter management and reproducible experiments.

### Available Configurations

- **`params/config.yaml`**: Balanced default settings for general use
- **`params/quick_config.yaml`**: Fast training/testing for development
- **`params/high_quality_config.yaml`**: Optimized settings for best results

### Configuration Structure

```yaml
model:
  name: "pointnet2_sem_seg"
  num_classes: 5
  npoint: 4096              # Points per sample
  block_size: 40.0         # Spatial block size

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "Adam"
  decay_rate: 1e-4
  step_size: 10
  lr_decay: 0.7

data:
  root_dir: "/path/to/data"
  val_ratio: 0.25          # Validation split ratio
  random_seed: 42          # For reproducible splits
  feat_group: "xyz"       # Feature group: xyz, p3, cap, n3
  num_workers: 10

testing:
  num_votes: 3             # Voting rounds for accuracy
  visual: true             # Save visualizations
  model_path: "./path/to/model.pth"
  output_dir: "/path/to/results"
```

### Usage Examples

```bash
# Quick development cycle
python train_semseg_mangrove3d.py --config params/quick_config.yaml --epoch 5

# High-quality training run
python train_semseg_mangrove3d.py --config params/high_quality_config.yaml

# Custom parameters with YAML base
python train_semseg_mangrove3d.py --learning_rate 0.002 --batch_size 16
```

### Command Line Override Priority

1. **Command line arguments** (highest priority)
2. **Custom config file** (if specified with --config)
3. **Default config file** (params/config.yaml)

## Data Features and Model Architecture

### Supported Feature Groups

The system supports multiple feature combinations for point cloud representation:

- **`xyz`**: X, Y, Z, intensity_adjusted, range_adjusted, z_adjusted
- **`p3`**: X, Y, Z, PCA1, PCA2, PCA3  
- **`cap`**: X, Y, Z, curvature, anisotropy, planarity
- **`n3`**: X, Y, Z, Pseudo-Rn, Pseudo-Gn, Pseudo-Bn

### PointNet++ Architecture

The implementation uses PointNet++ with the following specifications:

- **Hierarchical Feature Learning**: Multi-scale feature extraction through set abstraction layers
- **Set Abstraction Layers**: Robust to point density variations and spatial irregularities
- **Feature Propagation**: Upsampling layers for dense point-wise prediction
- **Multi-class Output**: 5-class semantic segmentation (Ground, Stem, Canopy, Roots, Objects)

## Training and Evaluation

### Data Loading Options

**Single Folder with Random Split (Recommended)**
```python
dataset = Mangrove3DDataset(
    data_root='/path/to/data/train_val',
    split='train',
    val_ratio=0.25,      # 25% for validation
    random_seed=42,      # Reproducible splits
    feat_group='xyz'    # Feature selection
)
```

**Separate Folders**
```python
dataset = Mangrove3DDataset(
    data_root='/path/to/data',
    split='train',       # Uses data/train/ folder
    feat_group='xyz'
)
```

### Evaluation Metrics

The system provides comprehensive evaluation:

- **Overall Accuracy**: Percentage of correctly classified points
- **Mean Class Accuracy**: Average accuracy across all classes
- **Mean IoU**: Mean Intersection over Union across classes
- **Per-class IoU**: Individual class performance analysis
- **Frequency Weighted IoU**: IoU weighted by class frequency
- **Dice Coefficient**: Additional similarity metric for validation

### Output and Visualization

Testing generates:

- **Prediction CSV**: Point coordinates with predicted class colors for visualization
- **Evaluation Report**: Comprehensive performance statistics and confusion matrix
- **Per-class Analysis**: Detailed breakdown of individual class performance
- **Visual Results**: Optional 3D visualization files (if `visual: true` in config)

## Project Structure

```
Pointnet_Pointnet2_pytorch/
├── data_utils/
│   └── Mangrove3DDataLoader.py     # Enhanced data loading with random splits
├── models/
│   ├── pointnet2_sem_seg.py        # PointNet++ model architecture
│   └── pointnet2_utils.py          # Model utility functions
├── params/
│   ├── config.yaml                 # Default configuration
│   ├── quick_config.yaml           # Fast training setup
│   ├── high_quality_config.yaml    # High-quality results setup
│   └── config_loader.py            # Configuration management utilities
├── train_semseg_mangrove3d.py      # Main training script
├── test_semseg_mangrove3d.py       # Main testing script
├── tools.py                        # General utility functions
└── log/                            # Training logs and checkpoints
    └── sem_seg/
        └── [timestamp]/
            ├── checkpoints/
            └── logs/
```

## Troubleshooting

### Common Issues and Solutions

**Out of Memory Error**
```bash
# Solution: Reduce batch size and points per sample
python train_semseg_mangrove3d.py --batch_size 8 --npoint 2048
```

**Data Loading Error**
- Verify data path in `params/config.yaml`
- Ensure file naming follows pattern: `*_color.csv` and `*_refined.label`
- Check that `train_val` folder exists or separate `train`/`val` folders are present
- Validate CSV format: X, Y, Z columns plus feature columns

**Configuration Error**
- Validate YAML syntax (proper indentation, no tabs)
- Check parameter names match the schema in configuration files
- Ensure all required fields are present in config file

**Model Loading Error**
```bash
# Check model path exists
python test_semseg_mangrove3d.py --model_path ./checkpoints/model.pth

# Use absolute path if relative path fails
python test_semseg_mangrove3d.py --model_path /full/path/to/model.pth
```

**Poor Performance**
- Increase number of training epochs
- Use higher quality configuration: `params/high_quality_config.yaml`
- Ensure adequate validation data (check `val_ratio` setting)
- Verify data quality and class balance

### Debugging Tips

**Enable Verbose Logging**
```bash
python train_semseg_mangrove3d.py --verbose
```

**Test Configuration Loading**
```python
from params.config_loader import load_config
config = load_config('params/config.yaml')
print(config)
```

**Validate Data Loading**
```python
from data_utils.Mangrove3DDataLoader import Mangrove3DDataset
dataset = Mangrove3DDataset(data_root='path/to/data', split='train')
print(f"Dataset size: {len(dataset)}")
```

## Advanced Usage and Optimization

### Experiment Management

**Create Custom Configurations**
```bash
# Copy and modify base configuration
cp params/config.yaml params/experiment_1.yaml
# Edit params/experiment_1.yaml with your parameters

# Run experiment
python train_semseg_mangrove3d.py --config params/experiment_1.yaml
```

**Programmatic Configuration**
```python
from params.config_loader import load_config

# Load and modify configuration
config = load_config()
config.set('training.epochs', 100)
config.set('data.val_ratio', 0.3)
config.set('model.npoint', 8192)
config.save('params/custom_config.yaml')
```

### Performance Optimization

**For Fast Development**
- Use `params/quick_config.yaml`
- Reduce `npoint` to 2048 for faster training
- Set `sample_rate` to 0.5 for smaller datasets
- Use fewer epochs for initial testing

**For Best Results**
- Use `params/high_quality_config.yaml`
- Increase `npoint` to 8192 for more detail
- Use more `num_votes` during testing (5-10)
- Train for more epochs with learning rate scheduling

**Memory Optimization**
```bash
# Reduce memory usage
python train_semseg_mangrove3d.py --batch_size 8 --npoint 2048

# For systems with limited memory
python train_semseg_mangrove3d.py --batch_size 4 --num_workers 4
```

### Batch Processing and Evaluation

**Test Multiple Files**
```bash
# Test different files systematically
for i in {0..5}; do
    python test_semseg_mangrove3d.py --test_idx $i --output_dir results/test_$i
done
```

**Parameter Sweeps**
```bash
# Test different learning rates
for lr in 0.01 0.001 0.0001; do
    python train_semseg_mangrove3d.py --learning_rate $lr --log_dir logs/lr_$lr
done
```

## Contributing and Development

When extending the project:

1. **Follow Code Structure**: Maintain the modular design and clear separation of concerns
2. **Update Configurations**: Add new parameters to YAML configuration files with appropriate defaults
3. **Add Documentation**: Include comprehensive docstrings and update README for new features
4. **Test Thoroughly**: Validate new functionality with various data configurations
5. **Maintain Compatibility**: Ensure backward compatibility with existing scripts and data formats

### Development Workflow

1. **Setup Development Environment**
   ```bash
   # Use quick config for rapid iteration
   cp params/quick_config.yaml params/dev_config.yaml
   # Edit dev_config.yaml for your development needs
   ```

2. **Test Changes**
   ```bash
   # Quick validation
   python train_semseg_mangrove3d.py --config params/dev_config.yaml --epoch 1
   ```

3. **Validate Configuration System**
   ```python
   # Test config loading
   from params.config_loader import load_config
   config = load_config('params/dev_config.yaml')
   ```

## Citation and License

### Citation
If you use this code for Mangrove3D research, please cite:

```bibtex
@article{mangrove3d_pointnet2,
  title={PointNet++ for Mangrove3D Point Cloud Segmentation},
  author={Your Name},
  journal={Your Journal/Conference},
  year={2025}
}
```

### Original PointNet++ Citation
```bibtex
@article{Pytorch_Pointnet_Pointnet2,
  Author = {Xu Yan},
  Title = {Pointnet/Pointnet++ Pytorch},
  Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
  Year = {2019}
}
```

### License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Support

For questions, issues, or contributions:

1. **Check the troubleshooting section** above for common issues
2. **Review the configuration documentation** in `params/` directory
3. **Validate your setup** using the provided test scripts
4. **Open an issue** with detailed information about your problem and environment

This implementation provides a robust, configurable, and well-documented framework for Mangrove3D point cloud segmentation research.
