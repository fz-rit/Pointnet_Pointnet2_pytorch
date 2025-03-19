import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
class PaMa3DDataset(Dataset):
    def __init__(self, split='train', 
                 data_root='/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024', 
                 num_point=4096, block_size=40.0, sample_rate=1.0, num_class=6, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.num_class = num_class
        file_format = 'csv'
        if split == 'train':
            self.split_dir = Path(data_root) / 'train' / file_format
        else:
            self.split_dir = Path(data_root) / 'test' / file_format
        
        file_paths = list(self.split_dir.glob(f'*.{file_format}'))  # Gets all files and directories
        self.scans_split = sorted([f for f in file_paths if f.is_file()])

        self.scan_points, self.scan_labels = [], []
        self.scan_coord_min, self.scan_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(self.num_class)

        for scan_path in tqdm(self.scans_split, total=len(self.scans_split)):
            scan_data = pd.read_csv(scan_path, sep=',')  # pandas dataframe with 19 columns.
            points = scan_data.loc[:, ['X', 'Y', 'Z', 'r', 'g', 'b']].to_numpy() # xyzrgb, N*6
            labels = scan_data.loc[:, 'class_id'].to_numpy()  # N
            tmp, _ = np.histogram(labels, range(self.num_class+1))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.scan_points.append(points), self.scan_labels.append(labels)
            self.scan_coord_min.append(coord_min), self.scan_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        scan_idxs = []
        for index in range(len(self.scans_split)):
            scan_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.scan_idxs = np.array(scan_idxs)
        print("Totally {} samples in {} set.".format(len(self.scan_idxs), split))

    def __getitem__(self, idx):
        scan_idx = self.scan_idxs[idx]
        points = self.scan_points[scan_idx]   # N * 6, xyzrgb
        labels = self.scan_labels[scan_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.scan_coord_max[scan_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.scan_coord_max[scan_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.scan_coord_max[scan_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.scan_idxs)


class Pama3dTestDataset():
    # prepare to give prediction on each points
    def __init__(self, data_root, block_points=4096, split='test', stride=1, num_class=6, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = data_root
        self.split = split
        self.stride = stride * block_size # controls how much the window (block) shifts; <1 - overlap; =1 - adjacent; >1 - gap
        self.scene_points_num = []
        assert split in ['train', 'test']
        
        file_format = 'csv'
        if split == 'train':
            self.split_dir = Path(data_root) / 'train' / file_format
        else:
            self.split_dir = Path(data_root) / 'test' / file_format
        
        file_paths = list(self.split_dir.glob(f'*.{file_format}'))  # Gets all files and directories
        self.scans_split = sorted([f for f in file_paths if f.is_file()])

        self.scene_points_list = []
        self.semantic_labels_list = []
        self.scan_coord_min, self.scan_coord_max = [], []
        for scan_path in self.scans_split:
            data_df = pd.read_csv(scan_path, sep=',')
            points = data_df.loc[:, ['X', 'Y', 'Z']].to_numpy()
            self.scene_points_list.append(data_df.loc[:, ['X', 'Y', 'Z', 'r', 'g', 'b']].to_numpy())
            self.semantic_labels_list.append(data_df.loc[:, 'class_id'].to_numpy())
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.scan_coord_min.append(coord_min), self.scan_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(num_class)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_class+1))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_scan, label_scan, sample_weight, index_scan = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                start_x = coord_min[0] + index_x * self.stride
                end_x = min(start_x + self.block_size, coord_max[0])
                start_x = end_x - self.block_size
                start_y = coord_min[1] + index_y * self.stride
                end_y = min(start_y + self.block_size, coord_max[1])
                start_y = end_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= start_x - self.padding) & (points[:, 0] <= end_x + self.padding) & (points[:, 1] >= start_y - self.padding) & (
                                points[:, 1] <= end_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (start_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (start_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_scan = np.vstack([data_scan, data_batch]) if data_scan.size else data_batch
                label_scan = np.hstack([label_scan, label_batch]) if label_scan.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_scan.size else batch_weight
                index_scan = np.hstack([index_scan, point_idxs]) if index_scan.size else point_idxs
        data_scan = data_scan.reshape((-1, self.block_points, data_scan.shape[1]))
        label_scan = label_scan.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_scan = index_scan.reshape((-1, self.block_points))
        return data_scan, label_scan, sample_weight, index_scan

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024'
    num_point, block_size, num_class = 4096, 40, 6
    sample_rate = 0.1
    print(f"Sample rate: {sample_rate}")
    point_data = PaMa3DDataset(split='train', data_root=data_root, 
                               num_point=num_point, 
                               block_size=block_size, 
                               sample_rate=sample_rate, 
                               num_class=num_class,
                               transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()