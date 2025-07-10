#!/usr/bin/env python3
"""
Utility script to reorganize data from separate train/val folders into a single train_val folder.
This allows for better random splitting with controlled validation ratios.
"""

import shutil
from pathlib import Path
import argparse


def reorganize_data(data_root: str, dry_run: bool = True):
    """
    Reorganize data from train/ and val/ folders into a single train_val/ folder.
    
    Args:
        data_root: Root directory containing train/ and val/ folders
        dry_run: If True, only print what would be done without actually moving files
    """
    data_root = Path(data_root)
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    train_val_dir = data_root / 'train_val'
    
    # Check if source directories exist
    if not train_dir.exists():
        print(f"Warning: {train_dir} does not exist")
        return
    if not val_dir.exists():
        print(f"Warning: {val_dir} does not exist")
        return
    
    # Check if target directory already exists
    if train_val_dir.exists():
        print(f"Warning: {train_val_dir} already exists")
        response = input("Do you want to continue? This might overwrite existing files. (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Get all files
    train_files = list(train_dir.rglob('*'))
    val_files = list(val_dir.rglob('*'))
    
    print(f"Found {len(train_files)} files in {train_dir}")
    print(f"Found {len(val_files)} files in {val_dir}")
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print("The following operations would be performed:")
        print(f"1. Create directory: {train_val_dir}")
        print(f"2. Copy {len(train_files)} files from {train_dir}")
        print(f"3. Copy {len(val_files)} files from {val_dir}")
        print("\nRun with --execute to actually perform the operations.")
        return
    
    # Create target directory
    train_val_dir.mkdir(exist_ok=True)
    print(f"Created directory: {train_val_dir}")
    
    # Copy files from train directory
    print(f"\nCopying files from {train_dir}...")
    for file_path in train_files:
        if file_path.is_file():
            # Maintain relative directory structure
            rel_path = file_path.relative_to(train_dir)
            target_path = train_val_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_path)
            print(f"  Copied: {rel_path}")
    
    # Copy files from val directory
    print(f"\nCopying files from {val_dir}...")
    for file_path in val_files:
        if file_path.is_file():
            # Maintain relative directory structure
            rel_path = file_path.relative_to(val_dir)
            target_path = train_val_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_path)
            print(f"  Copied: {rel_path}")
    
    print(f"\n✅ Successfully reorganized data into {train_val_dir}")
    print(f"Total files copied: {len([f for f in train_files if f.is_file()]) + len([f for f in val_files if f.is_file()])}")
    
    print("\n📝 Next steps:")
    print("1. Update your training script to use the new random splitting approach")
    print("2. Set val_ratio parameter (e.g., 0.25 for 25% validation)")
    print("3. Use random_seed parameter for reproducible splits")


def main():
    parser = argparse.ArgumentParser(description="Reorganize train/val data into single folder")
    parser.add_argument('data_root', type=str, help='Root directory containing train/ and val/ folders')
    parser.add_argument('--execute', action='store_true', help='Actually perform the operations (default is dry run)')
    
    args = parser.parse_args()
    
    reorganize_data(args.data_root, dry_run=not args.execute)


if __name__ == '__main__':
    main()
