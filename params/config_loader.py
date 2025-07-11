"""
Configuration loader for Mangrove3D point cloud segmentation.
Simplified to use only YAML files for parameter management.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


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
    
    def copy(self):
        """Create a deep copy of the configuration."""
        import copy as copy_module
        new_config = Config.__new__(Config)
        new_config.config_path = self.config_path
        new_config.config = copy_module.deepcopy(self.config)
        return new_config


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
    print(f"Feature group: {config.get('data.feat_group')}")
