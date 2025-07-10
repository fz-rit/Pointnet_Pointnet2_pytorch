#!/usr/bin/env python3
"""
Demonstration script showing how to use the YAML configuration system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def demo_basic_usage():
    """Demonstrate basic configuration usage."""
    print("=== Basic Configuration Usage ===")
    
    from params.config_loader import load_config
    
    # Load default configuration
    config = load_config()
    print(f"✓ Loaded default config from: {config.config_path}")
    
    # Access parameters
    print(f"  Model: {config.get('model.name')}")
    print(f"  Classes: {config.get('model.num_classes')}")
    print(f"  Epochs: {config.get('training.epochs')}")
    print(f"  Batch size: {config.get('training.batch_size')}")
    
    # Load custom configuration
    try:
        custom_config = load_config('params/quick_config.yaml')
        print(f"✓ Loaded custom config: quick_config.yaml")
        print(f"  Quick config epochs: {custom_config.get('training.epochs')}")
        print(f"  Quick config points: {custom_config.get('model.npoint')}")
    except FileNotFoundError:
        print("! Custom config not found (this is expected if running from different directory)")

def demo_argument_parsing():
    """Demonstrate argument parsing with configuration."""
    print("\n=== Argument Parsing Demo ===")
    
    from params.config_loader import load_config, create_train_parser
    
    # Create parser with config defaults
    config = load_config()
    parser = create_train_parser(config)
    
    # Simulate command line arguments
    test_args = ['--epoch', '100', '--batch_size', '16', '--learning_rate', '0.002']
    args = parser.parse_args(test_args)
    
    print(f"✓ Parsed arguments: {test_args}")
    print(f"  Epochs: {args.epoch}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Update config with parsed arguments
    config.update_from_args(args)
    print("✓ Updated config with parsed arguments")
    print(f"  Config epochs: {config.get('training.epochs')}")
    print(f"  Config batch size: {config.get('training.batch_size')}")

def demo_config_modification():
    """Demonstrate programmatic config modification."""
    print("\n=== Configuration Modification Demo ===")
    
    from params.config_loader import load_config
    
    config = load_config()
    
    # Show original values
    print("Original values:")
    print(f"  Epochs: {config.get('training.epochs')}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
    print(f"  Model points: {config.get('model.npoint')}")
    
    # Modify configuration
    config.set('training.epochs', 150)
    config.set('training.learning_rate', 0.0008)
    config.set('model.npoint', 6144)
    
    print("\nModified values:")
    print(f"  Epochs: {config.get('training.epochs')}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
    print(f"  Model points: {config.get('model.npoint')}")
    
    # Save to temporary file (don't overwrite original)
    temp_path = '/tmp/demo_config.yaml'
    config.save(temp_path)
    print(f"✓ Saved modified config to: {temp_path}")

def demo_training_simulation():
    """Simulate training script usage."""
    print("\n=== Training Script Simulation ===")
    
    from params.config_loader import load_config, create_train_parser
    
    # Simulate command line: python train_semseg_mangrove3d.py --epoch 5 --batch_size 8
    config = load_config()
    parser = create_train_parser(config)
    args = parser.parse_args(['--epoch', '5', '--batch_size', '8'])
    config.update_from_args(args)
    
    print("Training configuration:")
    print(f"  Model: {config.get('model.name')}")
    print(f"  Points per sample: {config.get('model.npoint')}")
    print(f"  Batch size: {config.get('training.batch_size')}")
    print(f"  Epochs: {config.get('training.epochs')}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
    print(f"  Data root: {config.get('data.root_dir')}")
    print(f"  Validation ratio: {config.get('data.val_ratio')}")
    print(f"  GPU: {config.get('hardware.gpu')}")

def demo_testing_simulation():
    """Simulate testing script usage."""
    print("\n=== Testing Script Simulation ===")
    
    from params.config_loader import load_config, create_test_parser
    
    # Simulate command line: python test_semseg_mangrove3d.py --num_votes 7 --test_idx 2
    config = load_config()
    parser = create_test_parser(config)
    args = parser.parse_args(['--num_votes', '7', '--test_idx', '2'])
    config.update_from_args(args)
    
    print("Testing configuration:")
    print(f"  Model path: {config.get('testing.model_path')}")
    print(f"  Test file index: {config.get('testing.test_idx')}")
    print(f"  Number of votes: {config.get('testing.num_votes')}")
    print(f"  Batch size: {config.get('testing.batch_size')}")
    print(f"  Block points: {config.get('testing.block_points')}")
    print(f"  Output dir: {config.get('testing.output_dir')}")
    print(f"  Visualization: {config.get('testing.visual')}")

if __name__ == "__main__":
    print("YAML Configuration System Demo")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        demo_argument_parsing()
        demo_config_modification()
        demo_training_simulation()
        demo_testing_simulation()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Edit params/config.yaml to set your default parameters")
        print("2. Create custom config files for different experiments")
        print("3. Use the updated train/test scripts with --config flag")
        print("4. Override parameters from command line as needed")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
