import argparse
import datetime
import logging
import numpy as np
import os
import shutil
import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
from tools import save_and_plot_loss_accuracy
import provider
from data_utils.Mangrove3DDataLoader import Mangrove3DDataset
from params.config_loader import load_config, create_train_parser
from config_utils import parse_train_args
from common_utils import (
    setup_environment, create_model, apply_model_optimizations, 
    load_checkpoint, save_checkpoint, setup_basic_logging,
    create_optimizer, log_experiment_info
)
import math


def parse_args():
    """Parse command line arguments using YAML configuration."""
    return parse_train_args()


def setup_logging(log_dir: Path, model_name: str):
    """Setup logging configuration."""
    return setup_basic_logging(log_dir / f'{model_name}.txt', "Model")


def setup_directories(args, config):
    """Setup experiment directories."""
    from common_utils import get_experiment_dir
    
    exp_dir = get_experiment_dir(config)
    exp_dir.mkdir(exist_ok=True)
    
    dirs = {
        'experiment': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    return dirs


def create_data_loaders(args, config):
    """Create training and validation data loaders."""
    dataset_config = {
        'data_root': config.get('data.root_dir'),
        'num_point': config.get('model.npoint'),
        'block_size': config.get('model.block_size'),
        'sample_rate': config.get('training.sample_rate'),
        'num_class': config.get('model.num_classes'),
        'transform': None,
        'feat_group': config.get('data.feat_group', 'xyz'),  # Default to 'xyz'
        'val_ratio': config.get('data.val_ratio'),
        'random_seed': config.get('data.random_seed')
    }
    
    train_dataset = Mangrove3DDataset(split='train', **dataset_config)
    val_dataset = Mangrove3DDataset(split='val', **dataset_config)
    
    loader_config = {
        'batch_size': config.get('training.batch_size'),
        'num_workers': config.get('data.num_workers'),
        'pin_memory': config.get('hardware.pin_memory'),
        'drop_last': config.get('hardware.drop_last')
    }
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, 
        worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
        **loader_config
    )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_config)
    
    return train_loader, val_loader, train_dataset.labelweights


def setup_model_and_optimizer(args, config):
    """Setup model, criterion, and optimizer."""
    classifier, criterion = create_model(config, for_training=True)
    
    # Apply optimizations
    apply_model_optimizations(classifier)
    
    # Try loading pretrained model
    start_epoch = 0
    try:
        model_path = Path('log/sem_seg/pointnet2_sem_seg/checkpoints/best_model_xxxxx.pth')
        print(f"Trying to load pretrained model: {model_path}")
        start_epoch = load_checkpoint(classifier, model_path, for_training=True)
    except:
        print('No pretrained model found, starting from scratch...')
        # Weight initialization is handled in load_checkpoint when it fails
    
    # Setup optimizer
    optimizer = create_optimizer(classifier, config)
    
    return classifier, criterion, optimizer, start_epoch


def train_epoch(classifier, criterion, optimizer, train_loader, weights, args, config, logger):
    """Train for one epoch."""
    classifier.train()
    total_correct = total_seen = loss_sum = 0
    num_classes = config.get('model.num_classes')
    
    for points, target in tqdm(train_loader, desc="Training", smoothing=0.9):
        optimizer.zero_grad()
        
        # Data augmentation
        points = points.data.numpy()
        points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
        points = torch.Tensor(points).float().cuda()
        target = target.long().cuda()
        points = points.transpose(2, 1)
        
        # Forward pass
        seg_pred, trans_feat = classifier(points)
        seg_pred = seg_pred.contiguous().view(-1, num_classes)
        target_flat = target.view(-1, 1)[:, 0]
        
        # Loss and backprop
        loss = criterion(seg_pred, target_flat, trans_feat, weights)
        loss.backward()
        optimizer.step()
        
        # Statistics
        pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
        batch_label = target_flat.cpu().data.numpy()
        correct = np.sum(pred_choice == batch_label)
        
        total_correct += correct
        total_seen += config.get('training.batch_size') * config.get('model.npoint')
        loss_sum += loss
    
    train_acc = total_correct / float(total_seen)
    train_loss = loss_sum / len(train_loader)
    
    logger.info(f'Training loss: {train_loss:.6f}, accuracy: {train_acc:.6f}')
    return train_loss, train_acc


def validate_epoch(classifier, criterion, val_loader, weights, args, config, logger):
    """Validate for one epoch."""
    classifier.eval()
    total_correct = total_seen = loss_sum = 0
    num_classes = config.get('model.num_classes')
    class_names = config.get('classes.names')
    
    total_seen_class = [0] * num_classes
    total_correct_class = [0] * num_classes
    total_iou_deno_class = [0] * num_classes
    
    with torch.no_grad():
        for points, target in tqdm(val_loader, desc="Validation", smoothing=0.9):
            points = torch.Tensor(points).float().cuda()
            target = target.long().cuda()
            points = points.transpose(2, 1)
            
            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            
            batch_label = target.cpu().data.numpy()
            target_flat = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target_flat, trans_feat, weights)
            loss_sum += loss
            
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += config.get('training.batch_size') * config.get('model.npoint')
            
            # Per-class statistics
            for l in range(num_classes):
                total_seen_class[l] += np.sum(batch_label == l)
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))
    
    # Calculate metrics
    val_loss = loss_sum / len(val_loader)
    val_acc = total_correct / float(total_seen)
    class_acc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
    mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
    
    # Log per-class IoU
    iou_log = "------- Per-Class IoU -------\n"
    for l in range(num_classes):
        class_iou = total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6)
        iou_log += f"{class_names[l]}: {class_iou:.4f}\n"
    
    logger.info(f'Validation loss: {val_loss:.6f}, accuracy: {val_acc:.6f}')
    logger.info(f'Class avg accuracy: {class_acc:.6f}, mIoU: {mIoU:.6f}')
    logger.info(iou_log)
    
    return val_loss, val_acc, mIoU


# def update_learning_rate(optimizer, epoch, config):
#     """Update learning rate with decay."""
#     learning_rate = config.get('training.learning_rate')
#     lr_decay = config.get('training.lr_decay')
#     step_size = config.get('training.step_size')
    
#     lr = max(learning_rate * (lr_decay ** (epoch // step_size)), 1e-5)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


def update_learning_rate(optimizer, epoch, config):
    """Update learning rate with cosine annealing."""
    initial_lr = config.get('training.learning_rate')  # η_max
    min_lr = config.get('training.min_learning_rate', 1e-5)  # η_min
    total_epochs = config.get('training.epochs', 100)  # T_max
    
    # Cosine annealing formula
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def update_bn_momentum(classifier, epoch, config):
    """Update BatchNorm momentum."""
    step_size = config.get('training.step_size')
    momentum = max(0.1 * (0.5 ** (epoch // step_size)), 0.01)
    
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.momentum = momentum
    
    classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
    return momentum



def save_checkpoint_wrapper(classifier, optimizer, epoch, mIoU, save_path, logger):
    """Save model checkpoint using common utility."""
    metrics = {'class_avg_iou': mIoU}
    save_checkpoint(classifier, optimizer, epoch, metrics, save_path, logger)


def main(args, config):
    """Main training function with block size sensitivity support."""
    # Setup environment
    setup_environment(config.get('hardware.gpu'))
    
    # Check if block_size is a list (sensitivity testing) or single value
    block_size_param = config.get('model.block_size')
    
    if isinstance(block_size_param, list):
        # Multiple block sizes - sensitivity testing
        print(f"\n{'='*80}")
        print(f"BLOCK SIZE SENSITIVITY TESTING")
        print(f"Testing block sizes: {block_size_param}")
        print(f"{'='*80}")
        
        results = {}
        model_paths = {}
        
        for block_size in block_size_param:
            print(f"\n{'='*60}")
            print(f"Training with block size: {block_size:.1f}m")
            print(f"{'='*60}")
            
            try:
                best_iou, model_path = train_single_block_size(args, config, block_size)
                results[block_size] = best_iou
                model_paths[block_size] = model_path
                
                print(f"✓ Block size {block_size:.1f}m completed. Best mIoU: {best_iou:.6f}")
                
            except Exception as e:
                print(f"✗ Block size {block_size:.1f}m failed: {str(e)}")
                results[block_size] = None
                model_paths[block_size] = None
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BLOCK SIZE SENSITIVITY RESULTS")
        print(f"{'='*80}")
        print(f"{'Block Size (m)':<15} {'Best mIoU':<12} {'Model Path'}")
        print("-" * 80)
        
        best_overall_block_size = None
        best_overall_iou = 0
        
        for block_size in block_size_param:
            iou = results[block_size]
            model_path = model_paths[block_size]
            
            if iou is not None:
                print(f"{block_size:<15.1f} {iou:<12.6f} {model_path}")
                if iou > best_overall_iou:
                    best_overall_iou = iou
                    best_overall_block_size = block_size
            else:
                print(f"{block_size:<15.1f} {'FAILED':<12} {'N/A'}")
        
        print("-" * 80)
        if best_overall_block_size is not None:
            print(f"Best overall: Block size {best_overall_block_size:.1f}m with mIoU {best_overall_iou:.6f}")
        else:
            print("All block size experiments failed!")
        
        # Save results summary
        import json
        summary_path = Path('./log/sem_seg/block_size_sensitivity_results.json')
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            'block_sizes': block_size_param,
            'results': {str(k): v for k, v in results.items()},
            'model_paths': {str(k): str(v) if v else None for k, v in model_paths.items()},
            'best_block_size': best_overall_block_size,
            'best_iou': best_overall_iou,
            'feat_group': config.get('data.feat_group'),
            'npoint': config.get('model.npoint'),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nResults summary saved to: {summary_path}")
        
    else:
        # Single block size - original behavior
        print(f"Training single model with block size: {block_size_param:.1f}m")
        train_single_block_size(args, config, block_size_param)

def train_single_block_size(args, config, block_size):
    """Train a model with a specific block size."""
    from common_utils import get_experiment_dir, get_model_name_with_block_size
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL WITH BLOCK SIZE: {block_size:.1f}m")
    print(f"{'='*80}")
    
    # Create a modified config for this specific block size
    config_copy = config.copy()
    config_copy.set('model.block_size', block_size)
    
    # Setup directories for this specific block size
    exp_dir = get_experiment_dir(config_copy, block_size)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    dirs = {
        'experiment': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(dirs['logs'], config_copy.get('model.name'))
    
    def log_string(s):
        logger.info(s)
        print(s)
    
    # Log experiment info
    log_experiment_info(args, config_copy, logger)
    
    # Log training setup
    logger.info("=" * 60)
    logger.info("TRAINING SETUP")
    logger.info("=" * 60)
    logger.info(f"Block size: {block_size:.1f}m")
    logger.info(f"Experiment directory: {dirs['experiment']}")
    logger.info(f"Checkpoints will be saved to: {dirs['checkpoints']}")
    logger.info("=" * 60)
    
    # Copy model files
    model_name = config_copy.get('model.name')
    shutil.copy(f'models/{model_name}.py', dirs['experiment'])
    shutil.copy('models/pointnet2_utils.py', dirs['experiment'])
    
    # Create data loaders
    log_string("Loading datasets...")
    train_loader, val_loader, labelweights = create_data_loaders(args, config_copy)
    weights = torch.Tensor(labelweights).cuda()
    
    log_string(f"Training samples: {len(train_loader.dataset)}")
    log_string(f"Validation samples: {len(val_loader.dataset)}")
    
    # Setup model
    classifier, criterion, optimizer, start_epoch = setup_model_and_optimizer(args, config_copy)
    
    # Training loop
    best_iou = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    feature_group = config_copy.get('data.feat_group', 'xyz')
    
    num_epochs = config_copy.get('training.epochs')
    for epoch in range(start_epoch, num_epochs):
        log_string(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Update learning rate and momentum
        lr = update_learning_rate(optimizer, epoch, config_copy)
        momentum = update_bn_momentum(classifier, epoch, config_copy)
        log_string(f'Learning rate: {lr:.6f}, BN momentum: {momentum:.6f}')
        
        # Train and validate
        train_loss, train_acc = train_epoch(classifier, criterion, optimizer, train_loader, weights, args, config_copy, logger)
        val_loss, val_acc, mIoU = validate_epoch(classifier, criterion, val_loader, weights, args, config_copy, logger)
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if mIoU >= best_iou:
            best_iou = mIoU
            model_name = get_model_name_with_block_size(config_copy, block_size)
            save_path = dirs['checkpoints'] / model_name
            
            save_checkpoint(classifier, optimizer, epoch, 
                          {'mIoU': mIoU, 'accuracy': val_acc}, save_path, logger)
            log_string(f'Best model saved with mIoU: {best_iou:.6f}')
        
        # Save checkpoint at intervals
        if (epoch + 1) % config_copy.get('logging.save_interval') == 0:
            checkpoint_path = dirs['checkpoints'] / f'checkpoint_epoch_{epoch+1:03d}_blk{block_size:.1f}.pth'
            save_checkpoint(classifier, optimizer, epoch, 
                          {'mIoU': mIoU, 'accuracy': val_acc}, checkpoint_path, logger)
    
    # Save final plots
    save_and_plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, dirs['experiment'])
    
    log_string(f'Training completed for block size {block_size:.1f}m')
    log_string(f'Best mIoU achieved: {best_iou:.6f}')
    
    return best_iou, dirs['checkpoints'] / get_model_name_with_block_size(config_copy, block_size)

if __name__ == '__main__':
    args, config = parse_args()
    main(args, config)
