"""
Configuration and argument parsing utilities for training and testing.
"""

from typing import Tuple
import argparse
from params.config_loader import load_config, create_train_parser, create_test_parser
from common_utils import auto_configure_testing_paths


def parse_train_args() -> Tuple[argparse.Namespace, object]:
    """Parse command line arguments for training using YAML configuration."""
    # Load default configuration
    config = load_config()
    
    # Create parser with config defaults
    parser = create_train_parser(config)
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.config:
        config = load_config(args.config)
        config.update_from_args(args)
    else:
        config.update_from_args(args)
    
    # Add config object to args for easy access
    args.config = config
    
    return args, config


def parse_test_args() -> Tuple[argparse.Namespace, object]:
    """Parse command line arguments for testing using YAML configuration."""
    # Load default configuration
    config = load_config()
    
    # Create parser with config defaults
    parser = create_test_parser(config)
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.config:
        config = load_config(args.config)
        config.update_from_args(args)
    else:
        config.update_from_args(args)
    
    # Auto-configure testing paths based on training settings
    auto_configure_testing_paths(config)
    
    return args, config
