# Block Size Sensitivity Testing

This guide explains how to use the new block size sensitivity testing feature to find the optimal block size for your point cloud segmentation model.

## Overview

The block size parameter controls the spatial sampling area during training and testing. With this feature, you can:
- Train multiple models with different block sizes automatically
- Test all models and compare their performance
- Find the optimal block size for your specific dataset

## Configuration

### Single Block Size (Original Behavior)
```yaml
model:
  block_size: 10.0  # Single value: trains one model with 10×10m blocks
```

### Multiple Block Sizes (Sensitivity Testing)
```yaml
model:
  block_size: [2.0, 5.0, 10.0, 20.0, 40.0]  # List: trains 5 models with different block sizes
```

## Usage

### 1. Training with Block Size Sensitivity

```bash
# Edit your config file to use a list of block sizes
python train_semseg_mangrove3d.py --config params/block_size_sensitivity_config.yaml
```

This will:
- Train a separate model for each block size
- Save models with names like `best_model_xyzi0_blk2.0.pth`, `best_model_xyzi0_blk5.0.pth`, etc.
- Create separate experiment directories: `log/sem_seg/xyzi0_blk2.0/`, `log/sem_seg/xyzi0_blk5.0/`, etc.
- Save a summary file: `log/sem_seg/block_size_sensitivity_results.json`

### 2. Testing with Block Size Sensitivity

```bash
# Use the same config file to test all trained models
python test_semseg_mangrove3d.py --config params/block_size_sensitivity_config.yaml
```

This will:
- Test each trained model with its corresponding block size
- Save results for each block size in separate directories
- Generate a comparison summary: `log/sem_seg/block_size_test_results.json`

## Understanding Block Sizes

| Block Size | Area | Use Case | Point Density (4096 pts) |
|------------|------|----------|---------------------------|
| 2.0m | 4 m² | Very detailed, high-res features | 1,024 pts/m² |
| 5.0m | 25 m² | Local features, individual trees | 164 pts/m² |
| 10.0m | 100 m² | Balanced context and detail | 41 pts/m² |
| 20.0m | 400 m² | Larger ecological patterns | 10 pts/m² |
| 40.0m | 1,600 m² | Landscape-scale features | 2.6 pts/m² |

## Model Naming Convention

Models are automatically named with block size information:
- `best_model_{feat_group}_blk{block_size:.1f}.pth`
- Examples:
  - `best_model_xyzi0_blk2.0.pth`
  - `best_model_xyzi0_blk10.0.pth`
  - `best_model_xyz_p3_blk20.0.pth`

## Directory Structure

```
log/sem_seg/
├── xyzi0_blk2.0/
│   ├── checkpoints/
│   │   └── best_model_xyzi0_blk2.0.pth
│   └── logs/
├── xyzi0_blk5.0/
│   ├── checkpoints/
│   │   └── best_model_xyzi0_blk5.0.pth
│   └── logs/
├── block_size_sensitivity_results.json
└── block_size_test_results.json
```

## Results Analysis

### Training Results (`block_size_sensitivity_results.json`)
```json
{
  "block_sizes": [2.0, 5.0, 10.0],
  "results": {
    "2.0": 0.6234,
    "5.0": 0.7156,
    "10.0": 0.6892
  },
  "best_block_size": 5.0,
  "best_iou": 0.7156
}
```

### Test Results (`block_size_test_results.json`)
```json
{
  "train_results": {"2.0": 0.6234, "5.0": 0.7156, "10.0": 0.6892},
  "test_results": {"2.0": 0.6108, "5.0": 0.7023, "10.0": 0.6745},
  "best_train_block_size": 5.0,
  "best_test_block_size": 5.0,
  "best_test_iou": 0.7023
}
```

## Tips for Block Size Selection

1. **For dense point clouds**: Start with smaller block sizes (2-10m)
2. **For sparse point clouds**: Use larger block sizes (10-40m)
3. **For small areas (<50m×50m)**: Use block sizes ≤10m
4. **For large areas (>100m×100m)**: Consider block sizes 20-40m
5. **For detailed segmentation**: Prefer higher point densities (>50 pts/m²)

## Example Configurations

### Quick Test (3 block sizes)
```yaml
model:
  block_size: [5.0, 10.0, 20.0]
  
training:
  epochs: 10  # Faster for testing
```

### Comprehensive Test (5 block sizes)
```yaml
model:
  block_size: [2.0, 5.0, 10.0, 20.0, 40.0]
  
training:
  epochs: 30  # More thorough training
```

### High-Resolution Test
```yaml
model:
  block_size: [1.0, 2.0, 5.0]
  npoint: 8192  # More points for detailed analysis
```

## Command Line Usage

```bash
# Train with sensitivity testing
python train_semseg_mangrove3d.py --config params/block_size_sensitivity_config.yaml

# Test with sensitivity testing
python test_semseg_mangrove3d.py --config params/block_size_sensitivity_config.yaml

# Override specific parameters
python train_semseg_mangrove3d.py --config params/block_size_sensitivity_config.yaml --epochs 15

# Train single block size (original behavior)
python train_semseg_mangrove3d.py --block_size 10.0
```

This feature helps you systematically find the optimal block size for your specific mangrove point cloud data!
