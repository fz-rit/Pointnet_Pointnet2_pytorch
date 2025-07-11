"""
Configuration utilities for training and testing.
Simplified to use only YAML configuration files.
"""

import argparse
from typing import Optional
from params.config_loader import load_config
from common_utils import auto_configure_testing_paths


def parse_config_args():
    """Parse command line arguments for config file."""
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='params/config.yaml',
                        help='Path to config file (default: params/config.yaml)')
    args = parser.parse_args()
    return args.config


def load_train_config(config_path: Optional[str] = None):
    """Load training configuration from YAML file."""
    if config_path is None:
        # Parse command line arguments
        config_path = parse_config_args()
    
    return load_config(config_path)


def load_test_config(config_path: Optional[str] = None):
    """Load testing configuration from YAML file with auto-configured paths."""
    if config_path is None:
        # Parse command line arguments
        config_path = parse_config_args()
    
    config = load_config(config_path)
    
    # Auto-configure testing paths based on training settings
    auto_configure_testing_paths(config)
    
    return config
