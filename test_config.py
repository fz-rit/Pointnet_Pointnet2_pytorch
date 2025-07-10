#!/usr/bin/env python3
"""
Test script to verify YAML configuration loading works correctly.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_config_loading():
    """Test configuration loading functionality."""
    try:
        from params.config_loader import load_config, create_train_parser, create_test_parser
        
        # Test basic config loading
        print("Testing configuration loading...")
        config = load_config()
        
        print("✓ Configuration loaded successfully!")
        print(f"  Model name: {config.get('model.name')}")
        print(f"  Number of classes: {config.get('model.num_classes')}")
        print(f"  Training epochs: {config.get('training.epochs')}")
        print(f"  Data root: {config.get('data.root_dir')}")
        print(f"  Validation ratio: {config.get('data.val_ratio')}")
        
        # Test train parser
        print("\nTesting train argument parser...")
        train_parser = create_train_parser(config)
        train_args = train_parser.parse_args(['--epoch', '10', '--batch_size', '16'])
        print("✓ Train parser created successfully!")
        print(f"  Parsed epochs: {train_args.epoch}")
        print(f"  Parsed batch size: {train_args.batch_size}")
        
        # Test test parser
        print("\nTesting test argument parser...")
        test_parser = create_test_parser(config)
        test_args = test_parser.parse_args(['--num_votes', '5'])
        print("✓ Test parser created successfully!")
        print(f"  Parsed num_votes: {test_args.num_votes}")
        
        # Test config update from args
        print("\nTesting config update from arguments...")
        config.update_from_args(train_args)
        print("✓ Config updated successfully!")
        print(f"  Updated epochs: {config.get('training.epochs')}")
        print(f"  Updated batch size: {config.get('training.batch_size')}")
        
        # Test class configuration
        print("\nTesting class configuration...")
        class_names = config.get('classes.names')
        class_colors = config.get('classes.colors')
        print(f"  Classes: {class_names}")
        print(f"  Colors shape: {len(class_colors)} x {len(class_colors[0]) if class_colors else 0}")
        
        print("\n🎉 All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_script_imports():
    """Test that the updated scripts can be imported without errors."""
    try:
        print("\nTesting script imports...")
        
        # Test train script imports
        print("  Testing train script...")
        sys.path.append('.')
        
        # Test individual functions instead of full import to avoid execution
        from train_semseg_mangrove3d import parse_args as train_parse_args
        print("    ✓ Train script parse_args imported")
        
        from train_semseg_mangrove3d import setup_logging, setup_directories
        print("    ✓ Train script utility functions imported")
        
        # Test test script imports
        print("  Testing test script...")
        from test_semseg_mangrove3d import parse_args as test_parse_args
        print("    ✓ Test script parse_args imported")
        
        from test_semseg_mangrove3d import setup_logging as test_setup_logging
        print("    ✓ Test script utility functions imported")
        
        print("✓ All script imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Script import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("YAML Configuration Test Suite")
    print("=" * 60)
    
    # Run tests
    config_test = test_config_loading()
    import_test = test_script_imports()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration Loading: {'✓ PASS' if config_test else '❌ FAIL'}")
    print(f"Script Imports: {'✓ PASS' if import_test else '❌ FAIL'}")
    
    if config_test and import_test:
        print("\n🎉 All tests passed! YAML configuration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
