"""
Configuration utilities for training and testing.
Simplified to use only YAML configuration files.
"""

from typing import Optional
from params.config_loader import load_config
from common_utils import auto_configure_testing_paths


def load_train_config(config_path: Optional[str] = None):
    """Load training configuration from YAML file."""
    if config_path is None:
        # Use default config path
        config_path = 'params/config.yaml'
    
    return load_config(config_path)


def load_test_config(config_path: Optional[str] = None):
    """Load testing configuration from YAML file with auto-configured paths."""
    if config_path is None:
        # Use default test config path
        config_path = 'params/config.yaml'
    
    config = load_config(config_path)
    
    # Auto-configure testing paths based on training settings
    auto_configure_testing_paths(config)
    
    return config
