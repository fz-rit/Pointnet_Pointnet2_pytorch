# Automatic Path Management for Mangrove3D Point Cloud Segmentation

## Overview

This update implements automatic path management for model checkpoints and test output directories, eliminating the need for manual path configuration in YAML files. The system automatically generates appropriate paths based on training settings.

## Features

### 1. Automatic Model Path Generation
- **Training**: Models are saved as `best_model_{feat_group}.pth` in the experiment's checkpoints directory
- **Testing**: The system automatically locates the best model based on the feature group and experiment directory

### 2. Automatic Output Directory Generation
- **Testing**: Results are automatically saved to `{data.root_dir}/test_results`
- **Directory Creation**: Output directories are created automatically if they don't exist

### 3. Experiment Directory Logic
- **With log_dir**: Uses the specified log directory
- **Without log_dir**: Uses `./log/sem_seg/{feat_group}` as the experiment directory

## Configuration Changes

### YAML Files Updated
All configuration files in `params/` folder have been updated:
- `config.yaml`
- `config_rc.yaml`
- `quick_config.yaml`

### Before (Manual Configuration)
```yaml
testing:
  model_path: "./log/sem_seg/2025-07-10/checkpoints/best_model_2025-07-10.pth"
  output_dir: "/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024/test_results"
```

### After (Automatic Configuration)
```yaml
testing:
  # model_path and output_dir are auto-generated based on training settings
  # model_path: auto-generated as {log_dir}/checkpoints/best_model_{feat_group}.pth
  # output_dir: auto-generated as {data.root_dir}/test_results
```

## Path Generation Rules

### Model Path
```
{experiment_dir}/checkpoints/best_model_{feat_group}.pth
```

Where `experiment_dir` is:
- `{logging.log_dir}` if specified
- `./log/sem_seg/{data.feat_group}` if log_dir is null

### Output Directory
```
{data.root_dir}/test_results
```

## Examples

### Example 1: Default Configuration
```yaml
data:
  root_dir: "/home/user/data/palau_2024"
  feat_group: "xyzi0"
logging:
  log_dir: null
```

**Generated Paths:**
- Model: `./log/sem_seg/xyzi0/checkpoints/best_model_xyzi0.pth`
- Output: `/home/user/data/palau_2024/test_results`

### Example 2: Custom Log Directory
```yaml
data:
  root_dir: "/home/user/data/palau_2024"
  feat_group: "xyz_irz"
logging:
  log_dir: "./experiments/custom_run"
```

**Generated Paths:**
- Model: `./experiments/custom_run/checkpoints/best_model_xyz_irz.pth`
- Output: `/home/user/data/palau_2024/test_results`

## New Utilities

### 1. Path Manager Script
```bash
# Show auto-generated paths for default config
python path_manager.py show

# Show paths for specific config
python path_manager.py show --config params/quick_config.yaml

# Create directories
python path_manager.py create --config params/config_rc.yaml
```

### 2. Common Utilities Functions
- `get_experiment_dir(config)`: Get experiment directory
- `get_best_model_path(config)`: Generate model path
- `get_test_output_dir(config)`: Generate output directory
- `auto_configure_testing_paths(config)`: Auto-configure all testing paths

## New Feature: Test All Files

### Overview
When `test_idx` is set to `-1`, the system will automatically run inference on **all available test files** in the dataset, providing comprehensive evaluation across the entire test set.

### Configuration
```yaml
testing:
  test_idx: -1  # Test all files (-1) or specific file index (>=0)
```

### Behavior Comparison

| test_idx Value | Behavior | Output Files |
|----------------|----------|--------------|
| `>= 0` | Test single file at specified index | `{idx:02d}_metrics.txt`, `{filename}_predictions.csv` |
| `-1` | Test ALL available files | Individual metrics + `overall_metrics.txt` + all visualizations |

### Output Structure for All Files Testing
```
test_results/
├── test_results.txt           # Complete execution log
├── 00_metrics.txt            # Metrics for file 0
├── 01_metrics.txt            # Metrics for file 1
├── ...                       # Metrics for each file
├── overall_metrics.txt       # Aggregated metrics across all files
├── file1_predictions.csv     # Visualizations (if enabled)
├── file2_predictions.csv
└── ...
```

### Aggregated Statistics
When testing all files, the system provides:
- **Overall metrics**: Computed across all points from all files
- **Per-file statistics**: Mean, standard deviation, min, max for each metric
- **Individual file listing**: Performance of each file for detailed analysis

### Example Usage
```bash
# Test all files with default config
python test_semseg_mangrove3d.py --test_idx -1

# Test all files with dedicated config
python test_semseg_mangrove3d.py --config params/test_all_config.yaml

# Test specific file
python test_semseg_mangrove3d.py --test_idx 2
```

### Log Output Example
```
Running inference on ALL 5 test files
Processing test file 1/5 (index 0)
...individual file results...
OVERALL RESULTS ACROSS ALL TEST FILES
Mean IoU: 0.8234
PER-FILE SUMMARY
Mean IoU - Average: 0.8234, Std: 0.0456
Individual file results:
  file1: mIoU=0.8456, Acc=0.9123
  file2: mIoU=0.8012, Acc=0.8987
```

## Benefits

1. **Consistency**: Ensures consistent naming across training and testing
2. **Automation**: No manual path updates needed when changing configurations
3. **Flexibility**: Supports both custom and automatic directory structures
4. **Safety**: Automatically creates directories to prevent errors
5. **Transparency**: Logs all auto-generated paths for verification

## Usage

### Training
```bash
python train_semseg_mangrove3d.py --config params/config.yaml
```
- Saves models to auto-generated experiment directory
- Logs experiment directory path

### Testing
```bash
python test_semseg_mangrove3d.py --config params/config.yaml
```
- Automatically finds the best model from training
- Saves results to auto-generated output directory
- Logs all auto-generated paths

## Backward Compatibility

- All existing functionality is preserved
- Manual path specification via command line arguments still works
- YAML files can still override paths if needed by uncommenting and modifying the relevant lines

## Logging

Both training and testing scripts now log the paths being used:

### Training Logs
```
TRAINING SETUP
==============
Experiment directory: ./log/sem_seg/xyzi0
Checkpoints will be saved to: ./log/sem_seg/xyzi0/checkpoints
```

### Testing Logs
```
AUTO-GENERATED PATHS
====================
Model path: ./log/sem_seg/xyzi0/checkpoints/best_model_xyzi0.pth
Output directory: /home/user/data/palau_2024/test_results
```

## File Structure

```
project/
├── common_utils.py          # Shared utility functions
├── config_utils.py          # Configuration parsing utilities  
├── path_manager.py          # Path management utility script
├── train_semseg_mangrove3d.py
├── test_semseg_mangrove3d.py
└── params/
    ├── config.yaml          # Updated with auto-path comments
    ├── config_rc.yaml       # Updated with auto-path comments
    └── quick_config.yaml    # Updated with auto-path comments
```

## Migration Guide

### For Existing Projects
1. **No action required** - paths are generated automatically
2. **Optional**: Remove hardcoded paths from custom YAML files
3. **Recommended**: Use `python path_manager.py show` to verify paths

### For New Projects
1. Configure `data.root_dir` and `data.feat_group` in YAML
2. Optionally set `logging.log_dir` for custom experiment directories
3. Run training and testing - paths are handled automatically
