import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any
import importlib

from data_utils.Mangrove3DDataLoader import Mangrove3DTestDataset
from tools import calc_metrics
from params.config_loader import load_config, create_test_parser

# Configuration
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any
import importlib

from data_utils.Mangrove3DDataLoader import Mangrove3DTestDataset
from tools import calc_metrics

# Configuration
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
CLASSES = ['Ground', 'Stem', 'Canopy', 'Roots', 'Objects']
COLOR_MAP = np.array([[128, 0, 128], [165, 42, 42], [0, 128, 0], [255, 165, 0], [255, 255, 0]])


def parse_args() -> Tuple[argparse.Namespace, object]:
    """Parse command line arguments using YAML configuration."""
    # Load default configuration
    config = load_config()
    
    # Create parser with config defaults
    parser = create_test_parser(config)
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.config:
        config = load_config(args.config)
        config.update_from_args(args)
    else:
        config.update_from_args(args)
    
    return args, config


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("TestModel")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(output_dir / 'test_results.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_model(model_path: Path, config) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    MODEL = importlib.import_module(config.get('model.name'))
    
    # Determine input channels based on feature group
    feat_group = config.get('data.feat_group', 'xyz')
    feature_map = {
        "xyz": 3,
        "xyzi0": 4,
        "xyz_irz": 6,
        "xyz_p3": 6,
        "xyz_cap": 6,
        "xyz_n3": 6
    }
    input_channels = feature_map.get(feat_group, 3)
    
    model = MODEL.get_model(config.get('model.num_classes'), input_channels=input_channels).cuda()
    
    assert model_path.exists(), f"Model not found: {model_path}"
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.eval()


def run_inference(model: torch.nn.Module, dataset: Mangrove3DTestDataset, test_idx: int, 
                 batch_size: int, num_votes: int, config, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a single test file with voting."""
    whole_scene_data = dataset.scene_points_list[test_idx]
    whole_scene_label = dataset.semantic_labels_list[test_idx]
    num_classes = config.get('model.num_classes')
    vote_pool = np.zeros((len(whole_scene_label), num_classes))
    
    logger.info(f"Processing file index {test_idx}, points: {len(whole_scene_label)}")
    
    with torch.no_grad():
        for vote in tqdm(range(num_votes), desc="Voting rounds"):
            scene_data, scene_label, scene_weights, scene_indices = dataset[test_idx]
            num_blocks = scene_data.shape[0]
            
            # Process in batches
            for start_idx in tqdm(range(0, num_blocks, batch_size), desc=f"Vote {vote+1}", leave=False):
                end_idx = min(start_idx + batch_size, num_blocks)
                batch_size_actual = end_idx - start_idx
                
                # Prepare batch data
                batch_data = np.zeros((batch_size, scene_data.shape[1], scene_data.shape[2]))
                batch_indices = np.zeros((batch_size, scene_data.shape[1]))
                batch_weights = np.zeros((batch_size, scene_data.shape[1]))
                
                batch_data[:batch_size_actual] = scene_data[start_idx:end_idx]
                batch_indices[:batch_size_actual] = scene_indices[start_idx:end_idx]
                batch_weights[:batch_size_actual] = scene_weights[start_idx:end_idx]
                
                # Run inference
                torch_data = torch.tensor(batch_data, dtype=torch.float32).cuda().transpose(2, 1)
                seg_pred, _ = model(torch_data)
                pred_labels = seg_pred.cpu().data.max(2)[1].numpy()
                
                # Add votes
                _add_votes(vote_pool, batch_indices[:batch_size_actual], 
                          pred_labels[:batch_size_actual], batch_weights[:batch_size_actual])
    
    return np.argmax(vote_pool, 1), whole_scene_data, whole_scene_label


def _add_votes(vote_pool: np.ndarray, point_idx: np.ndarray, pred_label: np.ndarray, weights: np.ndarray):
    """Add prediction votes to the voting pool."""
    batch_size, num_points = pred_label.shape
    for b in range(batch_size):
        for n in range(num_points):
            if weights[b, n] != 0 and not np.isinf(weights[b, n]):
                vote_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1


def evaluate_predictions(pred_labels: np.ndarray, gt_labels: np.ndarray, config, logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate predictions and log results."""
    num_classes = config.get('model.num_classes')
    class_names = config.get('classes.names')
    
    conf_mtx, overall_acc, mean_acc, mean_iou, fwiou, dice, class_ious = calc_metrics(
        gt_labels, pred_labels, num_classes
    )
    
    # Log results with cleaner formatting
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {overall_acc:.4f}")
    logger.info(f"Mean Class Accuracy: {mean_acc:.4f}")
    logger.info(f"Mean IoU: {mean_iou:.4f}")
    logger.info(f"Frequency Weighted IoU: {fwiou:.4f}")
    logger.info(f"Dice Coefficient: {dice:.4f}")
    logger.info("")
    logger.info("Per-Class IoU:")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name}: {class_ious[i]:.4f}")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(str(conf_mtx))
    
    return {
        'overall_accuracy': overall_acc,
        'mean_accuracy': mean_acc,
        'mean_iou': mean_iou,
        'fwiou': fwiou,
        'dice': dice,
        'class_ious': class_ious,
        'confusion_matrix': conf_mtx
    }


def save_visualization(scene_data: np.ndarray, pred_labels: np.ndarray, 
                      output_path: Path, config, logger: logging.Logger):
    """Save prediction visualization as CSV."""
    color_map = np.array(config.get('classes.colors'))
    pred_colors = color_map[pred_labels]
    
    vis_data = pd.DataFrame({
        'x': scene_data[:, 0],
        'y': scene_data[:, 1], 
        'z': scene_data[:, 2],
        'r': pred_colors[:, 0],
        'g': pred_colors[:, 1],
        'b': pred_colors[:, 2],
        'label': pred_labels + 1  # Convert back to 1-based for visualization
    })
    
    vis_data.to_csv(output_path, index=False)
    logger.info(f"Visualization saved to: {output_path}")


def main():
    """Main testing function."""
    start_time = time.time()
    args, config = parse_args()
    
    # Setup environment and directories
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get('hardware.gpu')
    output_dir = Path(config.get('testing.output_dir'))
    output_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Starting test with arguments: {vars(args)}")
    
    # Load dataset and model
    dataset = Mangrove3DTestDataset(
        data_root=config.get('data.root_dir'),
        split='test',
        block_points=config.get('testing.block_points'),
        num_class=config.get('model.num_classes')
    )
    logger.info(f"Loaded test dataset with {len(dataset)} files")
    
    model = load_model(Path(config.get('testing.model_path')), config)
    logger.info(f"Loaded model from: {config.get('testing.model_path')}")
    
    # Run inference
    pred_labels, scene_data, gt_labels = run_inference(
        model, dataset, config.get('testing.test_idx'), config.get('testing.batch_size'),
        config.get('testing.num_votes'), config, logger
    )
    
    # Evaluate and save results
    metrics = evaluate_predictions(pred_labels, gt_labels, config, logger)
    
    if config.get('testing.visual'):
        file_stem = Path(dataset.pcd_file_paths[config.get('testing.test_idx')]).stem
        output_path = output_dir / f"{file_stem}_predictions.csv"
        save_visualization(scene_data, pred_labels, output_path, config, logger)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Testing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final mIoU: {metrics['mean_iou']:.4f}")


if __name__ == '__main__':
    main()
