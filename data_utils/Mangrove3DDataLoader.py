import numpy as np
import pandas as pd
import torch
import time
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


def convert_labels(labels: np.ndarray) -> np.ndarray:
    """Convert labels from 1-based to 0-based indexing."""
    return labels - 1 if labels.min() == 1 else labels


def get_color_group(color_group: str) -> List[str]:
    """Get column names for different color feature groups."""
    color_groups = {
        "irz": ['X', 'Y', 'Z', 'intensity_adjusted', 'range_adjusted', 'z_adjusted'],
        "p3": ['X', 'Y', 'Z', 'PCA1', 'PCA2', 'PCA3'],
        "cap": ['X', 'Y', 'Z', 'curvature', 'anisotropy', 'planarity'],
        "n3": ['X', 'Y', 'Z', 'Pseudo-Rn', 'Pseudo-Gn', 'Pseudo-Bn']
    }
    if color_group not in color_groups:
        raise ValueError(f"Invalid color group '{color_group}'. Valid: {list(color_groups.keys())}")
    return color_groups[color_group]


class BaseMangrove3DDataset:
    """Base class for Mangrove3D datasets."""
    
    def __init__(self, data_root: str, split: str, num_class: int = 5, color_group: str = "irz", 
                 val_ratio: float = 0.25, random_seed: int = 42):
        self.data_root = Path(data_root)
        self.split = split
        self.num_class = num_class
        self.pts_col_names = get_color_group(color_group)
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Load and validate file paths
        self.pcd_file_paths, self.label_file_paths = self._load_and_split_files()
        
        assert len(self.pcd_file_paths) > 0, f"No PCD files found for split '{split}'"
        assert len(self.pcd_file_paths) == len(self.label_file_paths), \
            f"File count mismatch: {len(self.pcd_file_paths)} PCD vs {len(self.label_file_paths)} labels"
    
    def _load_and_split_files(self) -> Tuple[List[Path], List[Path]]:
        """Load files and split into train/val based on split parameter."""
        # First try the single folder approach (train_val)
        train_val_dir = self.data_root / 'train_val'
        if train_val_dir.exists():
            print(f"Using single folder approach: {train_val_dir}")
            all_pcd_paths = sorted(train_val_dir.rglob('*pcd_*_color.csv'))
            all_label_paths = sorted(train_val_dir.rglob('*pcd_*_refined.label'))
            
            # Create reproducible random split
            np.random.seed(self.random_seed)
            n_files = len(all_pcd_paths)
            indices = np.random.permutation(n_files)
            n_val = int(n_files * self.val_ratio)
            
            if self.split == 'train':
                selected_indices = indices[n_val:]
                print(f"Selected {len(selected_indices)} files for training ({1-self.val_ratio:.1%})")
            elif self.split == 'val':
                selected_indices = indices[:n_val]
                print(f"Selected {len(selected_indices)} files for validation ({self.val_ratio:.1%})")
            else:  # test
                # For test, look in separate test folder
                test_dir = self.data_root / 'test'
                if test_dir.exists():
                    return sorted(test_dir.rglob('*pcd_*_color.csv')), sorted(test_dir.rglob('*pcd_*_refined.label'))
                else:
                    raise ValueError(f"Test directory not found: {test_dir}")
            
            return ([all_pcd_paths[i] for i in selected_indices], 
                   [all_label_paths[i] for i in selected_indices])
        
        # Fallback to separate folders approach
        else:
            print(f"Using separate folders approach")
            split_dir = self.data_root / self.split
            if not split_dir.exists():
                raise ValueError(f"Split directory not found: {split_dir}")
            
            pcd_paths = sorted(split_dir.rglob('*pcd_*_color.csv'))
            label_paths = sorted(split_dir.rglob('*pcd_*_refined.label'))
            return pcd_paths, label_paths
    
    def load_points_labels(self, pcd_path: Path, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and validate points and labels."""
        points = pd.read_csv(pcd_path).loc[:, self.pts_col_names].to_numpy()
        labels = convert_labels(np.loadtxt(label_path, dtype=int))
        assert points.shape[0] == len(labels), f"Size mismatch: {points.shape[0]} vs {len(labels)}"
        return points, labels
    
    def compute_label_weights(self, all_labels: List[np.ndarray]) -> np.ndarray:
        """Compute balanced class weights."""
        counts = np.zeros(self.num_class)
        for labels in all_labels:
            hist, _ = np.histogram(labels, range(self.num_class + 1))
            counts += hist
        weights = counts.astype(np.float32) / np.sum(counts)
        return np.power(np.amax(weights) / weights, 1/3.0)

class Mangrove3DDataset(Dataset, BaseMangrove3DDataset):
    """Training dataset for Mangrove3D point cloud segmentation."""
    
    def __init__(self, split: str = 'train', data_root: str = None, 
                 num_point: int = 4096, block_size: float = 40.0, 
                 sample_rate: float = 1.0, num_class: int = 5, 
                 transform: Optional[callable] = None, color_group: str = "irz",
                 val_ratio: float = 0.25, random_seed: int = 42):
        Dataset.__init__(self)
        BaseMangrove3DDataset.__init__(self, data_root, split, num_class, color_group, val_ratio, random_seed)
        
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        
        # Load all data
        self.scan_points, self.scan_labels = [], []
        self.scan_coord_min, self.scan_coord_max = [], []
        
        for pcd_path, label_path in tqdm(zip(self.pcd_file_paths, self.label_file_paths), desc="Loading"):
            points, labels = self.load_points_labels(pcd_path, label_path)
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            
            self.scan_points.append(points)
            self.scan_labels.append(labels)
            self.scan_coord_min.append(coord_min)
            self.scan_coord_max.append(coord_max)
        
        self.labelweights = self.compute_label_weights(self.scan_labels)
        
        # Generate sampling indices
        num_points = [len(labels) for labels in self.scan_labels]
        sample_prob = np.array(num_points) / np.sum(num_points)
        num_iter = int(np.sum(num_points) * sample_rate / self.num_point)
        
        scan_idxs = []
        for i, prob in enumerate(sample_prob):
            scan_idxs.extend([i] * int(round(prob * num_iter)))
        self.scan_idxs = np.array(scan_idxs)
        
        print(f"Label weights: {self.labelweights}")
        print(f"Total {len(self.scan_idxs)} samples in {split} set")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        scan_idx = self.scan_idxs[idx]
        points = self.scan_points[scan_idx]
        labels = self.scan_labels[scan_idx]
        
        # Random block selection
        while True:
            center = points[np.random.choice(len(points))][:3]
            half_size = self.block_size / 2.0
            mask = ((points[:, 0] >= center[0] - half_size) & (points[:, 0] <= center[0] + half_size) &
                   (points[:, 1] >= center[1] - half_size) & (points[:, 1] <= center[1] + half_size))
            point_idxs = np.where(mask)[0]
            if len(point_idxs) > 1024:
                break
        
        # Sample points
        selected_idxs = np.random.choice(point_idxs, self.num_point, 
                                       replace=len(point_idxs) < self.num_point)
        selected_points = points[selected_idxs].copy()
        
        # Normalize features
        normalized_points = np.zeros((self.num_point, 9))
        coord_max = self.scan_coord_max[scan_idx]
        
        # Global normalization
        normalized_points[:, 6:9] = selected_points[:, :3] / coord_max
        
        # Center and normalize
        selected_points[:, :2] -= center[:2]
        selected_points[:, 3:6] /= 255.0
        normalized_points[:, :6] = selected_points
        
        current_labels = labels[selected_idxs]
        
        if self.transform:
            normalized_points, current_labels = self.transform(normalized_points, current_labels)
        
        return normalized_points, current_labels

    def __len__(self) -> int:
        return len(self.scan_idxs)


class Mangrove3DTestDataset(BaseMangrove3DDataset):
    """Test dataset with sliding window inference."""
    
    def __init__(self, data_root: str, block_points: int = 4096, split: str = 'test', 
                 stride: int = 1, num_class: int = 5, block_size: float = 1.0, 
                 padding: float = 0.001, color_group: str = "irz",
                 val_ratio: float = 0.25, random_seed: int = 42):
        super().__init__(data_root, split, num_class, color_group, val_ratio, random_seed)
        
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.stride = stride * block_size
        
        # Load test data
        self.scene_points_list = []
        self.semantic_labels_list = []
        
        for pcd_path, label_path in tqdm(zip(self.pcd_file_paths, self.label_file_paths), desc="Loading test"):
            points, labels = self.load_points_labels(pcd_path, label_path)
            self.scene_points_list.append(points)
            self.semantic_labels_list.append(labels)
        
        self.labelweights = self.compute_label_weights(self.semantic_labels_list)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        points = self.scene_points_list[index][:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        
        # Grid calculation
        grid_x = int(np.ceil((coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil((coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        
        data_batches, label_batches, weight_batches, index_batches = [], [], [], []
        
        for y_idx in range(grid_y):
            for x_idx in range(grid_x):
                # Window bounds
                start_x = coord_min[0] + x_idx * self.stride
                start_y = coord_min[1] + y_idx * self.stride
                end_x = min(start_x + self.block_size, coord_max[0])
                end_y = min(start_y + self.block_size, coord_max[1])
                start_x, start_y = end_x - self.block_size, end_y - self.block_size
                
                # Find points in window
                mask = ((points[:, 0] >= start_x - self.padding) & (points[:, 0] <= end_x + self.padding) &
                       (points[:, 1] >= start_y - self.padding) & (points[:, 1] <= end_y + self.padding))
                point_idxs = np.where(mask)[0]
                
                if len(point_idxs) == 0:
                    continue
                
                # Resample to block_points
                point_size = int(np.ceil(len(point_idxs) / self.block_points) * self.block_points)
                if point_size > len(point_idxs):
                    extra_idxs = np.random.choice(point_idxs, point_size - len(point_idxs), 
                                                replace=(point_size - len(point_idxs)) > len(point_idxs))
                    point_idxs = np.concatenate([point_idxs, extra_idxs])
                
                np.random.shuffle(point_idxs)
                
                # Process batch
                data_batch = points[point_idxs].copy()
                
                # Normalize and translate
                normalized_xyz = data_batch[:, :3] / coord_max
                data_batch[:, 0] -= (start_x + self.block_size / 2.0)
                data_batch[:, 1] -= (start_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate([data_batch, normalized_xyz], axis=1)
                
                label_batch = labels[point_idxs].astype(int)
                weight_batch = self.labelweights[label_batch]
                
                data_batches.append(data_batch)
                label_batches.append(label_batch)
                weight_batches.append(weight_batch)
                index_batches.append(point_idxs)
        
        if not data_batches:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Concatenate and reshape
        data_scan = np.vstack(data_batches).reshape((-1, self.block_points, data_batches[0].shape[1]))
        label_scan = np.hstack(label_batches).reshape((-1, self.block_points))
        weight_scan = np.hstack(weight_batches).reshape((-1, self.block_points))
        index_scan = np.hstack(index_batches).reshape((-1, self.block_points))
        
        return data_scan, label_scan, weight_scan, index_scan

    def __len__(self) -> int:
        return len(self.scene_points_list)

def main():
    """Example usage and testing."""
    data_root = '/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024/temp'
    config = {
        'num_point': 4096, 
        'block_size': 40, 
        'num_class': 5, 
        'sample_rate': 0.1, 
        'color_group': "irz",
        'val_ratio': 0.25,  # 25% for validation
        'random_seed': 42   # For reproducible splits
    }
    
    print(f"Sample rate: {config['sample_rate']}")
    print(f"Validation ratio: {config['val_ratio']}")
    
    # Test both train and validation datasets
    train_dataset = Mangrove3DDataset(split='train', data_root=data_root, **config)
    val_dataset = Mangrove3DDataset(split='val', data_root=data_root, **config)
    
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    # Test data loading
    train_data, train_labels = train_dataset[0]
    val_data, val_labels = val_dataset[0]
    print(f'Train sample shape: {train_data.shape}, {train_labels.shape}')
    print(f'Val sample shape: {val_data.shape}, {val_labels.shape}')
    
    # Test DataLoader performance with training dataset
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True,
        worker_init_fn=lambda w: random.seed(manual_seed + w)
    )
    
    print("\nTesting DataLoader performance...")
    for epoch in range(2):
        start_time = time.time()
        for i, (data, labels) in enumerate(train_loader):
            if i >= 3:
                break
            print(f'Epoch {epoch+1}, Batch {i+1}: {time.time() - start_time:.3f}s')
            start_time = time.time()


if __name__ == '__main__':
    main()