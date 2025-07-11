"""
Common utility functions for training and testing PointNet models.
This module contains shared functionality to avoid code duplication.
"""

import importlib
import logging
import numpy as np
import os
import sys
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Configuration
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Feature group mapping for input channels
FEATURE_MAP = {
    "xyz": 3,
    "xyzi0": 4,
    "xyz_irz": 6,
    "xyz_p3": 6,
    "xyz_cap": 6,
    "xyz_n3": 6
}


def get_input_channels(feat_group: str) -> int:
    """Get number of input channels based on feature group."""
    return FEATURE_MAP.get(feat_group, 3)


def setup_environment(gpu_id: str):
    """Setup CUDA environment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def create_model(config, for_training: bool = True) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Create model and criterion based on configuration.
    
    Args:
        config: Configuration object
        for_training: Whether this is for training (affects return type)
    
    Returns:
        model: The neural network model
        criterion: Loss function (None for testing)
    """
    MODEL = importlib.import_module(config.get('model.name'))
    num_classes = config.get('model.num_classes')
    
    # Determine input channels based on feature group
    feat_group = config.get('data.feat_group', 'xyz')
    # Handle list case - should be single value when creating model
    if isinstance(feat_group, list):
        feat_group = feat_group[0]
    
    input_channels = get_input_channels(feat_group)
    
    model = MODEL.get_model(num_classes, input_channels=input_channels).cuda()
    criterion = MODEL.get_loss().cuda() if for_training else None
    
    return model, criterion


def apply_model_optimizations(model: torch.nn.Module):
    """Apply common model optimizations."""
    # Apply inplace ReLU
    def inplace_relu(m):
        if 'ReLU' in m.__class__.__name__:
            m.inplace = True
    model.apply(inplace_relu)


def initialize_weights(model: torch.nn.Module):
    """Initialize model weights."""
    def weights_init(m):
        classname = m.__class__.__name__
        if 'Conv2d' in classname or 'Linear' in classname:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(weights_init)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, 
                   for_training: bool = False, logger: Optional[logging.Logger] = None) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        for_training: Whether loading for training (returns epoch) or testing
        logger: Optional logger for messages
    
    Returns:
        start_epoch: Starting epoch (0 if not training or no checkpoint)
    """
    start_epoch = 0
    
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if for_training and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            
        message = f"Loaded checkpoint: {checkpoint_path}"
        if for_training:
            message += f" (epoch {start_epoch})"
            
        if logger:
            logger.info(message)
        else:
            print(message)
            
    except Exception as e:
        message = f"Could not load checkpoint {checkpoint_path}: {e}"
        if for_training:
            message += " - Starting from scratch..."
            if logger:
                logger.warning(message)
            else:
                print(message)
            initialize_weights(model)
        else:
            # For testing, checkpoint loading failure is critical
            raise RuntimeError(message)
    
    return start_epoch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, metrics: Dict[str, float], save_path: Path, 
                   logger: Optional[logging.Logger] = None):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **metrics  # Include any metrics like mIoU, accuracy, etc.
    }
    
    torch.save(state, save_path)
    message = f"Model saved to {save_path}"
    
    if logger:
        logger.info(message)
    else:
        print(message)


def setup_basic_logging(log_file: Path, logger_name: str = "Model") -> logging.Logger:
    """Setup basic logging configuration."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_console_logging(logger: logging.Logger) -> logging.Logger:
    """Add console handler to existing logger."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    optimizer_name = config.get('training.optimizer')
    learning_rate = config.get('training.learning_rate')
    decay_rate = config.get('training.decay_rate')
    
    # Ensure decay_rate is a float (handle both string and numeric from YAML)
    if isinstance(decay_rate, str):
        decay_rate = float(decay_rate)
    
    # Ensure learning_rate is a float
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=0.9
        )
    
    return optimizer


def log_experiment_info(config, logger: logging.Logger):
    """Log experiment configuration and parameters."""
    logger.info('=' * 60)
    logger.info('EXPERIMENT CONFIGURATION')
    logger.info('=' * 60)
    logger.info(f'Model: {config.get("model.name")}')
    logger.info(f'Feature group: {config.get("data.feat_group")}')
    logger.info(f'Number of classes: {config.get("model.num_classes")}')
    logger.info(f'GPU: {config.get("hardware.gpu")}')
    logger.info('=' * 60)


def get_experiment_dir(config, block_size=None) -> Path:
    """
    Get the experiment directory based on logging configuration.
    
    Args:
        config: Configuration object
        block_size: Optional specific block size for naming (for sensitivity testing)
        
    Returns:
        Path to experiment directory
    """
    log_dir = config.get('logging.log_dir')
    feat_group = config.get('data.feat_group', 'xyz')
    
    # Normalize feat_group to string if it's a list
    if isinstance(feat_group, list):
        feat_group = feat_group[0]  # Use first one for directory naming
    
    # Handle block size for sensitivity testing
    if block_size is not None:
        block_suffix = f"_blk{block_size:.1f}"
    else:
        # Check if block_size is a list (sensitivity testing mode)
        model_block_size = config.get('model.block_size')
        if isinstance(model_block_size, list):
            block_suffix = "_sensitivity"
        else:
            block_suffix = f"_blk{model_block_size:.1f}"
    
    if log_dir:
        base_dir = Path(log_dir)
        return base_dir.parent / (base_dir.name + block_suffix) if block_size is not None or isinstance(model_block_size, list) else base_dir
    else:
        # Use feature group as directory name when log_dir is null
        return Path('./log/sem_seg') / (feat_group + block_suffix)


def get_best_model_path(config, block_size=None) -> Path:
    """
    Generate the best model path based on training configuration.
    
    Args:
        config: Configuration object
        block_size: Optional specific block size for naming (for sensitivity testing)
        
    Returns:
        Path to the best model checkpoint
    """
    exp_dir = get_experiment_dir(config, block_size)
    feat_group = config.get('data.feat_group', 'xyz')
    
    # Normalize feat_group to string if it's a list
    if isinstance(feat_group, list):
        feat_group = feat_group[0]
    
    if block_size is not None:
        model_name = f'best_model_{feat_group}_blk{block_size:.1f}.pth'
    else:
        # Check if block_size is a list (sensitivity testing mode)
        model_block_size = config.get('model.block_size')
        if isinstance(model_block_size, list):
            # For sensitivity testing, we can't auto-determine which model to use
            # This should be handled differently by the caller
            raise ValueError("Cannot auto-generate model path when block_size is a list. Use specific block_size parameter.")
        else:
            model_name = f'best_model_{feat_group}_blk{model_block_size:.1f}.pth'
    
    return exp_dir / 'checkpoints' / model_name


def get_test_output_dir(config) -> Path:
    """
    Generate the test output directory based on data configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Path to test results directory
    """
    root_dir = Path(config.get('data.root_dir'))
    return root_dir / 'test_results'


def auto_configure_testing_paths(config):
    """
    Automatically configure testing paths based on training settings.
    
    Args:
        config: Configuration object to update
    """
    # Get feature group (should be single value for testing individual models)
    feat_group = config.get('data.feat_group', 'xyz')
    if isinstance(feat_group, list):
        # Should have been set to single value by test_single_feat_group
        feat_group = feat_group[0] if feat_group else 'xyz'
    
    # Auto-generate model path
    exp_dir = get_experiment_dir(config)
    model_name = f'best_model_{feat_group}.pth'
    model_path = exp_dir / 'checkpoints' / model_name
    config.set('testing.model_path', str(model_path))
    
    # Auto-generate output directory
    output_dir = get_test_output_dir(config)
    config.set('testing.output_dir', str(output_dir))
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)


def get_model_name_with_block_size(config, block_size=None) -> str:
    """
    Generate model name with block size information for sensitivity testing.
    
    Args:
        config: Configuration object
        block_size: Optional specific block size for naming
        
    Returns:
        Model name string with block size suffix
    """
    feat_group = config.get('data.feat_group', 'xyz')
    
    if block_size is not None:
        return f"best_model_{feat_group}_blk{block_size:.1f}.pth"
    else:
        # Check if block_size is a list (this shouldn't happen in practice for single model)
        model_block_size = config.get('model.block_size')
        if isinstance(model_block_size, list):
            # This case should not happen when saving individual models
            raise ValueError("Cannot generate single model name when block_size is a list")
        else:
            return f"best_model_{feat_group}_blk{model_block_size:.1f}.pth"
