#!/usr/bin/env python3
"""
Demo script to test the new "test all files" functionality.
This script demonstrates the difference between testing a single file vs all files.
"""

import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("MANGROVE3D TEST ALL FILES FUNCTIONALITY DEMO")
    print("=" * 80)
    print()
    
    print("1. SINGLE FILE TESTING (Original behavior)")
    print("-" * 50)
    print("Config: test_idx: 0")
    print("Command: python test_semseg_mangrove3d.py --config params/config.yaml")
    print("Result: Tests only the file at index 0")
    print("Output files:")
    print("  - 00_metrics.txt (metrics for file 0)")
    print("  - {filename}_predictions.csv (if visual=true)")
    print()
    
    print("2. ALL FILES TESTING (New functionality)")
    print("-" * 50)
    print("Config: test_idx: -1")
    print("Command: python test_semseg_mangrove3d.py --config params/test_all_config.yaml")
    print("OR: python test_semseg_mangrove3d.py --test_idx -1")
    print("Result: Tests ALL available files in the dataset")
    print("Output files:")
    print("  - 00_metrics.txt, 01_metrics.txt, ... (metrics for each file)")
    print("  - overall_metrics.txt (aggregated metrics across all files)")
    print("  - {filename}_predictions.csv for each file (if visual=true)")
    print()
    
    print("3. SAMPLE COMMANDS")
    print("-" * 50)
    print("# Test single file (index 0)")
    print("python test_semseg_mangrove3d.py --config params/config.yaml --test_idx 0")
    print()
    print("# Test single file (index 2)")
    print("python test_semseg_mangrove3d.py --config params/config.yaml --test_idx 2")
    print()
    print("# Test ALL files")
    print("python test_semseg_mangrove3d.py --config params/config.yaml --test_idx -1")
    print()
    print("# Test ALL files with dedicated config")
    print("python test_semseg_mangrove3d.py --config params/test_all_config.yaml")
    print()
    
    print("4. OUTPUT STRUCTURE FOR ALL FILES TESTING")
    print("-" * 50)
    print("test_results/")
    print("├── test_results.txt           # Complete log")
    print("├── 00_metrics.txt            # Metrics for file 0")
    print("├── 01_metrics.txt            # Metrics for file 1")
    print("├── ...                       # Metrics for each file")
    print("├── overall_metrics.txt       # Aggregated metrics")
    print("├── file1_predictions.csv     # Predictions for file 1")
    print("├── file2_predictions.csv     # Predictions for file 2")
    print("└── ...                       # Predictions for each file")
    print()
    
    print("5. AGGREGATED METRICS INCLUDE")
    print("-" * 50)
    print("- Overall accuracy across all files")
    print("- Mean IoU across all files")
    print("- Per-class IoU across all files")
    print("- Statistics (mean, std, min, max) for individual file results")
    print("- List of individual file performance")
    print()
    
    print("6. LOG OUTPUT EXAMPLES")
    print("-" * 50)
    print("For test_idx = -1:")
    print("  'Running inference on ALL 5 test files'")
    print("  'Processing test file 1/5 (index 0)'")
    print("  '... individual file results ...'")
    print("  'OVERALL RESULTS ACROSS ALL TEST FILES'")
    print("  'PER-FILE SUMMARY'")
    print("  'Mean IoU - Average: 0.8234, Std: 0.0456'")
    print()
    print("For test_idx = 0:")
    print("  'Running inference on single test file (index 0)'")
    print("  '... single file results ...'")
    print()
    
    print("=" * 80)
    print("Ready to test! Use the commands above to try the new functionality.")
    print("=" * 80)

if __name__ == '__main__':
    main()
