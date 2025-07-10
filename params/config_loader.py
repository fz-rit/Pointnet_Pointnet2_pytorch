"""
Configuration loader for Mangrove3D point cloud segmentation.
"""
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import os


class Config:
    """Configuration class for managing parameters."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        assert self.config_path.exists(), f"Config file not found: {self.config_path}"
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'model.name')."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments."""
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None:  # Only update if argument was provided
                # Map common argument names to config paths
                key_mapping = {
                    'model': 'model.name',
                    'batch_size': 'training.batch_size',
                    'epoch': 'training.epochs',
                    'learning_rate': 'training.learning_rate',
                    'optimizer': 'training.optimizer',
                    'decay_rate': 'training.decay_rate',
                    'step_size': 'training.step_size',
                    'lr_decay': 'training.lr_decay',
                    'npoint': 'model.npoint',
                    'data_root': 'data.root_dir',
                    'val_ratio': 'data.val_ratio',
                    'random_seed': 'data.random_seed',
                    'gpu': 'hardware.gpu',
                    'log_dir': 'logging.log_dir',
                    'num_votes': 'testing.num_votes',
                    'test_idx': 'testing.test_idx',
                    'visual': 'testing.visual',
                    'model_path': 'testing.model_path',
                    'output_dir': 'testing.output_dir',
                    'block_points': 'testing.block_points'
                }
                
                config_key = key_mapping.get(key, key)
                self.set(config_key, value)


def create_train_parser(config: Config) -> argparse.ArgumentParser:
    """Create argument parser for training with defaults from config."""
    parser = argparse.ArgumentParser('Mangrove3D Semantic Segmentation Training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=config.get('model.name'), 
                       help='Model name')
    parser.add_argument('--npoint', type=int, default=config.get('model.npoint'), 
                       help='Number of points')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=config.get('training.batch_size'), 
                       help='Batch size')
    parser.add_argument('--epoch', type=int, default=config.get('training.epochs'), 
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=config.get('training.learning_rate'), 
                       help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default=config.get('training.optimizer'), 
                       help='Optimizer type')
    parser.add_argument('--decay_rate', type=float, default=config.get('training.decay_rate'), 
                       help='Weight decay rate')
    parser.add_argument('--step_size', type=int, default=config.get('training.step_size'), 
                       help='LR decay step size')
    parser.add_argument('--lr_decay', type=float, default=config.get('training.lr_decay'), 
                       help='LR decay rate')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default=config.get('data.root_dir'),
                       help='Data root directory')
    parser.add_argument('--val_ratio', type=float, default=config.get('data.val_ratio'),
                       help='Validation split ratio')
    parser.add_argument('--random_seed', type=int, default=config.get('data.random_seed'),
                       help='Random seed for reproducible splits')
    
    # Hardware arguments
    parser.add_argument('--gpu', type=str, default=config.get('hardware.gpu'), 
                       help='GPU to use')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default=config.get('logging.log_dir'), 
                       help='Log directory')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file (overrides default)')
    
    return parser


def create_test_parser(config: Config) -> argparse.ArgumentParser:
    """Create argument parser for testing with defaults from config."""
    parser = argparse.ArgumentParser('Mangrove3D Semantic Segmentation Testing')
    
    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=config.get('testing.batch_size'), 
                       help='Batch size for testing')
    parser.add_argument('--block_points', type=int, default=config.get('testing.block_points'), 
                       help='Points per block')
    parser.add_argument('--num_votes', type=int, default=config.get('testing.num_votes'), 
                       help='Number of voting rounds')
    parser.add_argument('--test_idx', type=int, default=config.get('testing.test_idx'), 
                       help='Index of test file to process')
    parser.add_argument('--visual', action='store_true', default=config.get('testing.visual'), 
                       help='Save visualization')
    parser.add_argument('--model_path', type=str, default=config.get('testing.model_path'),
                       help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default=config.get('testing.output_dir'),
                       help='Output directory')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default=config.get('data.root_dir'),
                       help='Data root directory')
    
    # Hardware arguments
    parser.add_argument('--gpu', type=str, default=config.get('hardware.gpu'), 
                       help='GPU device')
    
    # Config arguments
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file (overrides default)')
    
    return parser


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Model name: {config.get('model.name')}")
    print(f"Number of classes: {config.get('model.num_classes')}")
    print(f"Data root: {config.get('data.root_dir')}")
