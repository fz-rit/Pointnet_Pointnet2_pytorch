
"""
Summary:
This script collects input files for the Mangrove3D project from a specified root directory.
It organizes the files into train/val and test directories based on their parent directory names.
It copies pcd files and label files into corresponding directories.
Author: fzhcis
Date: 2025-06-23
"""

import shutil
from pathlib import Path


train_val_dirs = ['ALRSET2', 'ALRSET3', 'AURSET2', 'AURSET3', 'AURSET1']
test_dirs = ['ALRSET1', 'D1B375']
copy_root_dir = Path('/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024')
root_dir = Path('/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/palau_2024')

copy_test_dir = copy_root_dir / 'test'
copy_train_val_dir = copy_root_dir / 'train_val'


pcd_csv_files = list(root_dir.glob('**/outputs/pcd/pcd_*_color.csv'))
label_files = list(root_dir.glob('**/outputs/pcd/pcd_*_refined.label'))

pcd_csv_files.sort()
label_files.sort()

print(f'Found {len(pcd_csv_files)} pcd files, and {len(label_files)} label files.')

for i in range(39):
    pcd_file = pcd_csv_files[i]
    label_file = label_files[i]

    parent_dir = pcd_file.parent.parent.parent.parent
    assert label_file.parent.parent.parent.parent == parent_dir, \
        f'label file {label_file.name} does not match pcd parent directory {parent_dir}'

    if any(x in str(parent_dir) for x in test_dirs):
        copy_dir = copy_test_dir
    elif any(x in str(parent_dir) for x in train_val_dirs):
        copy_dir = copy_train_val_dir
    else:
        raise ValueError(f'Unknown parent directory: {parent_dir}')

    pcd_dest_dir = copy_dir / 'pcd' 
    label_dest_dir = copy_dir / 'label'

    pcd_dest_dir.mkdir(parents=True, exist_ok=True)
    label_dest_dir.mkdir(parents=True, exist_ok=True)
    pcd_dest = pcd_dest_dir / f"proj{i:03d}_{pcd_file.name}"
    label_dest = label_dest_dir / f"proj{i:03d}_{label_file.name}"


    shutil.copy2(pcd_file, pcd_dest)
    shutil.copy2(label_file, label_dest)
    print(f'Copy directory: {copy_dir}')
    print(f'Copied {pcd_file.name} to {pcd_dest.name}')
    print(f'Copied {label_file.name} to {label_dest.name}')