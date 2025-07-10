#!/usr/bin/env python3
"""
Path management utility for Mangrove3D point cloud segmentation.
This script demonstrates the automatic path generation for training and testing.
"""

import sys
import argparse
from pathlib import Path
from params.config_loader import load_config
from common_utils import get_experiment_dir, get_best_model_path, get_test_output_dir, auto_configure_testing_paths


def show_paths(config_file: str = None):
    """Show the auto-generated paths for a given configuration."""
    print("=" * 80)
    print("MANGROVE3D PATH MANAGEMENT UTILITY")
    print("=" * 80)
    
    # Load configuration
    config = load_config(config_file)
    
    print(f"Configuration file: {config.config_path}")
    print(f"Feature group: {config.get('data.feat_group')}")
    print(f"Data root: {config.get('data.root_dir')}")
    print(f"Log directory setting: {config.get('logging.log_dir')}")
    print()
    
    # Show training paths
    print("TRAINING PATHS:")
    print("-" * 40)
    exp_dir = get_experiment_dir(config)
    print(f"Experiment directory: {exp_dir}")
    print(f"Checkpoints directory: {exp_dir}/checkpoints")
    print(f"Logs directory: {exp_dir}/logs")
    print()
    
    # Show testing paths
    print("TESTING PATHS (Auto-generated):")
    print("-" * 40)
    model_path = get_best_model_path(config)
    output_dir = get_test_output_dir(config)
    
    print(f"Best model path: {model_path}")
    print(f"Test output directory: {output_dir}")
    print()
    
    # Show path existence
    print("PATH EXISTENCE CHECK:")
    print("-" * 40)
    print(f"Experiment directory exists: {exp_dir.exists()}")
    print(f"Checkpoints directory exists: {(exp_dir / 'checkpoints').exists()}")
    print(f"Best model exists: {model_path.exists()}")
    print(f"Test output directory exists: {output_dir.exists()}")
    print()
    
    # Apply auto-configuration
    print("AUTO-CONFIGURATION RESULT:")
    print("-" * 40)
    auto_configure_testing_paths(config)
    print(f"Config testing.model_path: {config.get('testing.model_path')}")
    print(f"Config testing.output_dir: {config.get('testing.output_dir')}")
    print(f"Test output directory created: {output_dir.exists()}")
    
    print("=" * 80)


def create_directories(config_file: str = None):
    """Create the necessary directories for training and testing."""
    config = load_config(config_file)
    
    print("Creating directories...")
    
    # Create training directories
    exp_dir = get_experiment_dir(config)
    (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'logs').mkdir(parents=True, exist_ok=True)
    print(f"✓ Created training directories under: {exp_dir}")
    
    # Create testing directory
    output_dir = get_test_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created test output directory: {output_dir}")
    
    print("All directories created successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Mangrove3D Path Management Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show paths for default config
  python path_manager.py show
  
  # Show paths for specific config
  python path_manager.py show --config params/quick_config.yaml
  
  # Create directories for default config
  python path_manager.py create
  
  # Create directories for specific config
  python path_manager.py create --config params/config_rc.yaml
        """
    )
    
    parser.add_argument('action', choices=['show', 'create'], 
                       help='Action to perform: show paths or create directories')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: params/config.yaml)')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'show':
            show_paths(args.config)
        elif args.action == 'create':
            create_directories(args.config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
