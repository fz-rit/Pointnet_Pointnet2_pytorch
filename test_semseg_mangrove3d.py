import argparse
import datetime
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

from data_utils.Mangrove3DDataLoader import Mangrove3DTestDataset
from tools import calc_metrics, write_eval_metrics_to_file
from params.config_loader import load_config, create_test_parser
from config_utils import parse_test_args
from common_utils import (
    setup_environment, create_model, load_checkpoint, 
    setup_basic_logging, setup_console_logging, log_experiment_info
)
# CLASSES = ['Ground', 'Stem', 'Canopy', 'Roots', 'Objects']
# COLOR_MAP = np.array([[128, 0, 128], [165, 42, 42], [0, 128, 0], [255, 165, 0], [255, 255, 0]])


def parse_args() -> Tuple[argparse.Namespace, object]:
    """Parse command line arguments using YAML configuration."""
    return parse_test_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    logger = setup_basic_logging(output_dir / 'test_results.txt', "TestModel")
    return setup_console_logging(logger)


def load_model(model_path: Path, config) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model, _ = create_model(config, for_training=False)
    load_checkpoint(model, model_path, for_training=False)
    return model.eval()


def run_inference_single(model: torch.nn.Module, dataset: Mangrove3DTestDataset, test_idx: int, 
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


def run_inference(model: torch.nn.Module, dataset: Mangrove3DTestDataset, test_idx: int, 
                 batch_size: int, num_votes: int, config, logger: logging.Logger, output_dir: Path) -> Dict[str, Any]:
    """
    Run inference on test dataset(s).
    
    Args:
        model: Trained model
        dataset: Test dataset
        test_idx: Test index (-1 for all files, >=0 for specific file)
        batch_size: Batch size for inference
        num_votes: Number of voting rounds
        config: Configuration object
        logger: Logger
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing aggregated metrics
    """
    if test_idx == -1:
        # Run inference on all available test files
        logger.info(f"Running inference on ALL {len(dataset)} test files")
        all_metrics = []
        overall_predictions = []
        overall_ground_truth = []
        
        for idx in range(len(dataset)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing test file {idx+1}/{len(dataset)} (index {idx})")
            logger.info(f"{'='*60}")
            
            # Run inference on single file
            pred_labels, scene_data, gt_labels = run_inference_single(
                model, dataset, idx, batch_size, num_votes, config, logger
            )
            
            # Evaluate single file
            metrics = evaluate_predictions(pred_labels, gt_labels, config, logger)
            all_metrics.append(metrics)
            
            # Save individual results
            write_eval_metrics_to_file(metrics, output_dir, f"{idx:02d}")
            
            # Save visualization if enabled
            if config.get('testing.visual'):
                file_stem = Path(dataset.pcd_file_paths[idx]).stem
                output_path = output_dir / f"{file_stem}_predictions.csv"
                save_visualization(scene_data, pred_labels, output_path, config, logger)
            
            # Accumulate for overall statistics
            overall_predictions.extend(pred_labels)
            overall_ground_truth.extend(gt_labels)
        
        # Calculate overall metrics across all files
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL RESULTS ACROSS ALL TEST FILES")
        logger.info(f"{'='*60}")
        
        overall_metrics = evaluate_predictions(
            np.array(overall_predictions), np.array(overall_ground_truth), config, logger
        )
        
        # Save overall metrics
        write_eval_metrics_to_file(overall_metrics, output_dir, "overall")
        
        # Calculate and log per-file statistics
        logger.info(f"\n{'='*60}")
        logger.info("PER-FILE SUMMARY")
        logger.info(f"{'='*60}")
        
        mean_ious = [m['mean_iou'] for m in all_metrics]
        overall_accs = [m['overall_accuracy'] for m in all_metrics]
        
        logger.info(f"Mean IoU - Average: {np.mean(mean_ious):.4f}, Std: {np.std(mean_ious):.4f}")
        logger.info(f"Mean IoU - Min: {np.min(mean_ious):.4f}, Max: {np.max(mean_ious):.4f}")
        logger.info(f"Overall Acc - Average: {np.mean(overall_accs):.4f}, Std: {np.std(overall_accs):.4f}")
        logger.info(f"Overall Acc - Min: {np.min(overall_accs):.4f}, Max: {np.max(overall_accs):.4f}")
        
        # Log individual file results
        logger.info("\nIndividual file results:")
        for idx, metrics in enumerate(all_metrics):
            file_stem = Path(dataset.pcd_file_paths[idx]).stem
            logger.info(f"  {file_stem}: mIoU={metrics['mean_iou']:.4f}, Acc={metrics['overall_accuracy']:.4f}")
        
        return overall_metrics
        
    else:
        # Run inference on single file (original behavior)
        logger.info(f"Running inference on single test file (index {test_idx})")
        pred_labels, scene_data, gt_labels = run_inference_single(
            model, dataset, test_idx, batch_size, num_votes, config, logger
        )
        
        # Evaluate and save results
        metrics = evaluate_predictions(pred_labels, gt_labels, config, logger)
        write_eval_metrics_to_file(metrics, output_dir, f"{test_idx:02d}")
        
        if config.get('testing.visual'):
            file_stem = Path(dataset.pcd_file_paths[test_idx]).stem
            output_path = output_dir / f"{file_stem}_predictions.csv"
            save_visualization(scene_data, pred_labels, output_path, config, logger)
        
        return metrics


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


def test_single_block_size(args, config, block_size, model_path):
    """Test a model with a specific block size."""
    print(f"\n{'='*80}")
    print(f"TESTING MODEL WITH BLOCK SIZE: {block_size:.1f}m")
    print(f"Model: {model_path}")
    print(f"{'='*80}")
    
    # Create a modified config for this specific block size
    config_copy = config.copy()
    config_copy.set('model.block_size', block_size)
    
    # Setup environment and directories
    setup_environment(config_copy.get('hardware.gpu'))
    
    # Create output directory for this block size
    base_output_dir = Path(config_copy.get('data.root_dir')) / 'test_results'
    output_dir = base_output_dir / f"blk{block_size:.1f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    log_experiment_info(args, config_copy, logger)
    
    # Log auto-generated paths
    logger.info("=" * 60)
    logger.info("BLOCK SIZE TESTING")
    logger.info("=" * 60)
    logger.info(f"Block size: {block_size:.1f}m")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Load dataset and model
    dataset = Mangrove3DTestDataset(
        data_root=config_copy.get('data.root_dir'),
        split='test',
        feat_group=config_copy.get('data.feat_group'),
        block_points=config_copy.get('testing.block_points'),
        num_class=config_copy.get('model.num_classes'),
        block_size=block_size  # Use the specific block size
    )
    logger.info(f"Loaded test dataset with {len(dataset)} files")
    
    if not Path(model_path).exists():
        error_msg = f"Model file not found: {model_path}"
        logger.error(error_msg)
        print(f"✗ {error_msg}")
        return None
    
    model = load_model(Path(model_path), config_copy)
    logger.info(f"Loaded model from: {model_path}")
    
    # Run inference
    metrics = run_inference(
        model, dataset, config_copy.get('testing.test_idx'), config_copy.get('testing.batch_size'),
        config_copy.get('testing.num_votes'), config_copy, logger, output_dir
    )
    
    logger.info(f"Testing completed for block size {block_size:.1f}m")
    logger.info(f"Final mIoU: {metrics['mean_iou']:.4f}")
    
    return metrics['mean_iou']


def main():
    """Main testing function with block size sensitivity support."""
    start_time = time.time()
    args, config = parse_args()
    
    # Check if block_size is a list (sensitivity testing) or single value
    block_size_param = config.get('model.block_size')
    
    if isinstance(block_size_param, list):
        # Multiple block sizes - sensitivity testing
        print(f"\n{'='*80}")
        print(f"BLOCK SIZE SENSITIVITY TESTING")
        print(f"Testing block sizes: {block_size_param}")
        print(f"{'='*80}")
        
        # Load the block size sensitivity results to find model paths
        results_path = Path('./log/sem_seg/block_size_sensitivity_results.json')
        if not results_path.exists():
            print(f"Error: Sensitivity results file not found: {results_path}")
            print("Please run training with block size sensitivity first.")
            return
        
        import json
        with open(results_path, 'r') as f:
            sensitivity_data = json.load(f)
        
        test_results = {}
        
        for block_size in block_size_param:
            block_size_str = str(block_size)
            if block_size_str in sensitivity_data['model_paths'] and sensitivity_data['model_paths'][block_size_str]:
                model_path = sensitivity_data['model_paths'][block_size_str]
                
                print(f"\n{'='*60}")
                print(f"Testing block size: {block_size:.1f}m")
                print(f"{'='*60}")
                
                try:
                    mIoU = test_single_block_size(args, config, block_size, model_path)
                    test_results[block_size] = mIoU
                    print(f"✓ Block size {block_size:.1f}m completed. mIoU: {mIoU:.6f}")
                    
                except Exception as e:
                    print(f"✗ Block size {block_size:.1f}m failed: {str(e)}")
                    test_results[block_size] = None
            else:
                print(f"✗ No trained model found for block size {block_size:.1f}m")
                test_results[block_size] = None
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BLOCK SIZE SENSITIVITY TEST RESULTS")
        print(f"{'='*80}")
        print(f"{'Block Size (m)':<15} {'Training mIoU':<15} {'Test mIoU':<12}")
        print("-" * 80)
        
        best_test_block_size = None
        best_test_iou = 0
        
        for block_size in block_size_param:
            train_iou = sensitivity_data['results'].get(str(block_size))
            test_iou = test_results[block_size]
            
            train_str = f"{train_iou:.6f}" if train_iou else "FAILED"
            test_str = f"{test_iou:.6f}" if test_iou else "FAILED"
            
            print(f"{block_size:<15.1f} {train_str:<15} {test_str:<12}")
            
            if test_iou is not None and test_iou > best_test_iou:
                best_test_iou = test_iou
                best_test_block_size = block_size
        
        print("-" * 80)
        if best_test_block_size is not None:
            print(f"Best test performance: Block size {best_test_block_size:.1f}m with mIoU {best_test_iou:.6f}")
        else:
            print("All block size tests failed!")
        
        # Save test results summary
        test_summary_path = Path('./log/sem_seg/block_size_test_results.json')
        test_summary_data = {
            'block_sizes': block_size_param,
            'train_results': sensitivity_data['results'],
            'test_results': {str(k): v for k, v in test_results.items()},
            'best_train_block_size': sensitivity_data.get('best_block_size'),
            'best_train_iou': sensitivity_data.get('best_iou'),
            'best_test_block_size': best_test_block_size,
            'best_test_iou': best_test_iou,
            'feat_group': config.get('data.feat_group'),
            'npoint': config.get('model.npoint'),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(test_summary_path, 'w') as f:
            json.dump(test_summary_data, f, indent=2)
        
        print(f"\nTest results summary saved to: {test_summary_path}")
        
    else:
        # Single block size - original behavior
        print(f"Testing single model with block size: {block_size_param:.1f}m")
        
        # Setup environment and directories
        setup_environment(config.get('hardware.gpu'))
        output_dir = Path(config.get('testing.output_dir'))
        output_dir.mkdir(exist_ok=True)
        
        logger = setup_logging(output_dir)
        log_experiment_info(args, config, logger)
        
        # Log auto-generated paths
        logger.info("=" * 60)
        logger.info("AUTO-GENERATED PATHS")
        logger.info("=" * 60)
        logger.info(f"Model path: {config.get('testing.model_path')}")
        logger.info(f"Output directory: {config.get('testing.output_dir')}")
        logger.info("=" * 60)
        
        # Load dataset and model
        dataset = Mangrove3DTestDataset(
            data_root=config.get('data.root_dir'),
            split='test',
            feat_group=config.get('data.feat_group'),
            block_points=config.get('testing.block_points'),
            num_class=config.get('model.num_classes'),
            block_size=config.get('model.block_size') 
        )
        logger.info(f"Loaded test dataset with {len(dataset)} files")
        
        model = load_model(Path(config.get('testing.model_path')), config)
        logger.info(f"Loaded model from: {config.get('testing.model_path')}")
        
        # Run inference
        metrics = run_inference(
            model, dataset, config.get('testing.test_idx'), config.get('testing.batch_size'),
            config.get('testing.num_votes'), config, logger, output_dir
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Testing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Final mIoU: {metrics['mean_iou']:.4f}")
        
        # Log summary based on test mode
        test_idx = config.get('testing.test_idx')
        if test_idx == -1:
            logger.info(f"Processed ALL {len(dataset)} test files")
            logger.info("Check individual file results in the output directory")
        else:
            logger.info(f"Processed single test file (index {test_idx})")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal testing time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()
