import argparse
import os
from data_utils.PaMa3DDataLoader import Pama3dTestDataset
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import pandas as pd
import time
from tools import calc_metrics
start_time = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


seg_label_to_cat = {0: 'Ground', 1: 'Stem', 2: 'Canopy', 3: 'Roots', 4: 'Objects'}


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--block_points', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='log/sem_seg', help='experiment root')
    parser.add_argument('--visual', action='store_true', default=True, help='visualize result [default: True]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # experiment_dir = Path(args.log_dir)
    experiment_dir = Path('log/sem_seg/2025-04-11_11-41')
    visual_dir = experiment_dir  / 'visual'
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(experiment_dir / 'eval.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 5
    BATCH_SIZE = args.batch_size
    BLOCK_POINTS = args.block_points

    data_root = '/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024_for_rc'

    TEST_DATASET = Pama3dTestDataset(split='test', data_root=data_root, block_points=BLOCK_POINTS, 
                                     num_class=NUM_CLASSES)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module('pointnet2_sem_seg')
    segmodel = MODEL.get_model(NUM_CLASSES).cuda()
    # model_path = experiment_dir / 'checkpoints' / 'model_2025-03-20_16-29.pth'
    model_path = Path('./log/sem_seg/2025-04-11_11-41/checkpoints/model_2025-04-11_11-41.pth')
    assert model_path.exists() and model_path.is_file(), f"Model checkpoint not found at {model_path}"
    checkpoint = torch.load(model_path, weights_only=False)
    segmodel.load_state_dict(checkpoint['model_state_dict'])
    segmodel = segmodel.eval()

    with torch.no_grad():
        test_img_idx = 3
        file_path = TEST_DATASET.scans_split[test_img_idx]
        file_stem = Path(file_path).stem
        print(f"Inference the file {file_stem}")
        

        whole_scene_data = TEST_DATASET.scene_points_list[test_img_idx]
        whole_scene_label = TEST_DATASET.semantic_labels_list[test_img_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        for _ in tqdm(range(args.num_votes), total=args.num_votes, desc="Votes", leave=False):
        
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET[test_img_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, BLOCK_POINTS, 9))
            batch_label = np.zeros((BATCH_SIZE, BLOCK_POINTS))
            batch_point_index = np.zeros((BATCH_SIZE, BLOCK_POINTS))
            batch_smpw = np.zeros((BATCH_SIZE, BLOCK_POINTS)) # sample weight

            for sbatch in tqdm(range(s_batch_num), total=s_batch_num, desc="Sub-batches", leave=False):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)
                seg_pred, _ = segmodel(torch_data)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                            batch_pred_label[0:real_batch_size, ...],
                                            batch_smpw[0:real_batch_size, ...])

        pred_label = np.argmax(vote_label_pool, 1)
        # print(pred_label.shape)
        # print(whole_scene_label.shape)
        # print(np.unique(pred_label))
        # print(np.unique(whole_scene_label))

        # Calculate metrics
        conf_mtx, overall_accuracy, mAcc, mIoU, FWIoU, dice_coefficient, IoUs = calc_metrics(whole_scene_label, pred_label, NUM_CLASSES)

        result_str = '------- Evaluation Results --------\n'
        result_str += 'Confusion Matrix:\n'
        result_str += f"{conf_mtx}\n" 
        result_str += 'Overall Accuracy: %.3f\n' % overall_accuracy
        result_str += 'Mean Class Accuracy: %.3f\n' % mAcc
        result_str += 'Mean IoU: %.3f\n' % mIoU
        result_str += 'Frequency Weighted IoU: %.3f\n' % FWIoU
        result_str += 'Dice Coefficient: %.3f\n' % dice_coefficient
        result_str += 'IoU for each class: \n'
        for i in range(NUM_CLASSES):
            result_str += f"{seg_label_to_cat[i]}: {IoUs[i]:.3f}\n"
        log_string(result_str)
        output_csv = visual_dir  / f"{file_stem}_pred.csv"
        if args.visual:
            # Save the prediction label and color to csv file
            color_map = np.array([[128, 0, 128], [165, 42, 42], [0, 128, 0], [255, 165, 0], [255, 255, 0]])
            pred_colors = color_map[pred_label]

            pred_data = pd.DataFrame({
                "x": whole_scene_data[:, 0],
                "y": whole_scene_data[:, 1],
                "z": whole_scene_data[:, 2],
                "r": pred_colors[:, 0],
                "g": pred_colors[:, 1],
                "b": pred_colors[:, 2],
                "label": pred_label+1
            })

            pred_data.to_csv(output_csv, index=False)
            log_string(f"Save the prediction to {output_csv}")

        print("--- Done with %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    args = parse_args()
    main(args)
