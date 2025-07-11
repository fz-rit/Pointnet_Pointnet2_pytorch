# Feature Group Sensitivity Testing

This update extends the `feat_group` setting to support testing multiple feature combinations automatically.

## Configuration

The `feat_group` parameter in the config file now accepts:
- **Single string**: `"xyzi0"` (original behavior)
- **List of strings**: `["xyz", "xyzi0", "xyz_p3"]` (sensitivity testing)

### Available Feature Groups
- `xyz`: X, Y, Z coordinates (3 channels)
- `xyzi0`: X, Y, Z coordinates + intensity (4 channels)
- `xyz_irz`: X, Y, Z + intensity + range + zenith (6 channels)
- `xyz_p3`: X, Y, Z + RGB colors (6 channels)
- `xyz_cap`: X, Y, Z + RGB colors from canopy analysis (6 channels)
- `xyz_n3`: X, Y, Z + surface normals (6 channels)

## Usage

### Single Feature Group (Original Behavior)
```yaml
data:
  feat_group: "xyzi0"  # Train/test single model
```

```bash
python train_semseg_mangrove3d.py --config params/config.yaml
python test_semseg_mangrove3d.py --config params/config.yaml
```

### Multiple Feature Groups (Sensitivity Testing)
```yaml
data:
  feat_group: ["xyz", "xyzi0", "xyz_p3"]  # Test multiple feature groups
```

```bash
# Train models for all feature groups
python train_semseg_mangrove3d.py --config params/feat_group_sensitivity.yaml

# Test all trained models
python test_semseg_mangrove3d.py --config params/feat_group_sensitivity.yaml
```

## Output Structure

### Single Feature Group
```
log/sem_seg/xyzi0_blk2.0/
├── checkpoints/
│   └── best_model_xyzi0.pth
├── logs/
└── experiment_info.txt
```

### Multiple Feature Groups
```
log/sem_seg/
├── xyz_blk2.0/
│   ├── checkpoints/best_model_xyz.pth
│   └── logs/
├── xyzi0_blk2.0/
│   ├── checkpoints/best_model_xyzi0.pth
│   └── logs/
└── xyz_p3_blk2.0/
    ├── checkpoints/best_model_xyz_p3.pth
    └── logs/
```

## Results Summary

When running sensitivity testing, the scripts will:

1. **Training**: Train separate models for each feature group and show a summary:
```
================================================================================
FEATURE GROUP SENSITIVITY RESULTS
================================================================================
Feature Group   Best mIoU
------------------------------
xyz             0.456789
xyzi0           0.567890  ← BEST
xyz_p3          0.445678

Best feature group: xyzi0 with mIoU 0.567890
```

2. **Testing**: Test all trained models and show results:
```
================================================================================
FEATURE GROUP TEST RESULTS
================================================================================
Feature Group   mIoU
-------------------------
xyz             0.445123
xyzi0           0.556789  ← BEST
xyz_p3          0.434567

Best feature group: xyzi0 with mIoU 0.556789
```

## Example Config Files

- `params/config.yaml`: Single feature group configuration
- `params/feat_group_sensitivity.yaml`: Multiple feature group configuration

## Implementation Notes

- Each feature group is trained independently with its own directory structure
- Model paths are automatically generated based on feature group names
- The data loader handles different input channel counts automatically
- Testing scripts can find and load the appropriate trained models
- Results are backward compatible with existing single feature group workflows
