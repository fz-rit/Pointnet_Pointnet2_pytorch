
import argparse
import datetime
import importlib
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

# Configuration
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))


def parse_args():
    """Parse command line arguments using YAML configuration."""
    # Load default configuration
    config = load_config()
    
    # Create parser with config defaults
    parser = create_train_parser(config)
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.config:
        config = load_config(args.config)
        config.update_from_args(args)
    else:
        config.update_from_args(args)
    
    # Add config object to args for easy access
    args.config = config
    
    return args, config


def setup_logging(log_dir: Path, model_name: str):
    """Setup logging configuration."""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_dir / f'{model_name}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_directories(args):
    """Setup experiment directories."""
    timestr = datetime.datetime.now().strftime('%Y-%m-%d')
    exp_dir = Path(args.log_dir) if args.log_dir else Path('./log/sem_seg/') / timestr
    exp_dir.mkdir(exist_ok=True)
    
    dirs = {
        'experiment': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    return dirs, timestr


def create_data_loaders(args, config):
    """Create training and validation data loaders."""
    dataset_config = {
        'data_root': config.get('data.root_dir'),
        'num_point': config.get('model.npoint'),
        'block_size': config.get('model.block_size'),
        'sample_rate': config.get('training.sample_rate'),
        'num_class': config.get('model.num_classes'),
        'transform': None,
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
    MODEL = importlib.import_module(config.get('model.name'))
    num_classes = config.get('model.num_classes')
    
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
    
    classifier = MODEL.get_model(num_classes, input_channels=input_channels).cuda()
    criterion = MODEL.get_loss().cuda()
    
    # Apply inplace ReLU
    def inplace_relu(m):
        if 'ReLU' in m.__class__.__name__:
            m.inplace = True
    classifier.apply(inplace_relu)
    
    # Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if 'Conv2d' in classname or 'Linear' in classname:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    # Try loading pretrained model
    start_epoch = 0
    try:
        model_path = Path('log/sem_seg/pointnet2_sem_seg/checkpoints/best_model_xxxxx.pth')
        print(f"Trying to load pretrained model: {model_path}")
        checkpoint = torch.load(model_path, weights_only=False)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded pretrained model: {model_path}')
    except:
        print('No pretrained model found, starting from scratch...')
        classifier.apply(weights_init)
    
    # Setup optimizer
    optimizer_name = config.get('training.optimizer')
    learning_rate = config.get('training.learning_rate')
    decay_rate = config.get('training.decay_rate')
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9)
    
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


def update_learning_rate(optimizer, epoch, config):
    """Update learning rate with decay."""
    learning_rate = config.get('training.learning_rate')
    lr_decay = config.get('training.lr_decay')
    step_size = config.get('training.step_size')
    
    lr = max(learning_rate * (lr_decay ** (epoch // step_size)), 1e-5)
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


def save_checkpoint(classifier, optimizer, epoch, mIoU, save_path, logger):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'class_avg_iou': mIoU,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, save_path)
    logger.info(f'Model saved to {save_path}')


def main(args, config):
    """Main training function."""
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get('hardware.gpu')
    
    # Setup directories and logging
    dirs, timestr = setup_directories(args)
    logger = setup_logging(dirs['logs'], config.get('model.name'))
    
    def log_string(s):
        logger.info(s)
        print(s)
    
    log_string('PARAMETERS:')
    log_string(str(args))
    
    # Copy model files
    model_name = config.get('model.name')
    shutil.copy(f'models/{model_name}.py', dirs['experiment'])
    shutil.copy('models/pointnet2_utils.py', dirs['experiment'])
    
    # Create data loaders
    log_string("Loading datasets...")
    train_loader, val_loader, labelweights = create_data_loaders(args, config)
    weights = torch.Tensor(labelweights).cuda()
    
    log_string(f"Training samples: {len(train_loader.dataset)}")
    log_string(f"Validation samples: {len(val_loader.dataset)}")
    
    # Setup model
    classifier, criterion, optimizer, start_epoch = setup_model_and_optimizer(args, config)
    
    # Training loop
    best_iou = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    num_epochs = config.get('training.epochs')
    for epoch in range(start_epoch, num_epochs):
        log_string(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Update learning rate and momentum
        lr = update_learning_rate(optimizer, epoch, config)
        momentum = update_bn_momentum(classifier, epoch, config)
        log_string(f'Learning rate: {lr:.6f}, BN momentum: {momentum:.3f}')
        
        # Train and validate
        train_loss, train_acc = train_epoch(classifier, criterion, optimizer, train_loader, weights, args, config, logger)
        val_loss, val_acc, mIoU = validate_epoch(classifier, criterion, val_loader, weights, args, config, logger)

        # Log losses and accuracies (convert tensors to float for plotting)
        train_losses.append(float(train_loss.cpu()) if hasattr(train_loss, 'cpu') else train_loss)
        val_losses.append(float(val_loss.cpu()) if hasattr(val_loss, 'cpu') else val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save checkpoint periodically
        save_interval = config.get('logging.save_interval')
        if epoch % save_interval == 0:
            save_path = dirs['checkpoints'] / f'model_{timestr}_epoch_{epoch}.pth'
            save_checkpoint(classifier, optimizer, epoch, mIoU, save_path, logger)
        
        # Save best model
        if mIoU >= best_iou:
            best_iou = mIoU
            save_path = dirs['checkpoints'] / f'best_model_{timestr}.pth'
            save_checkpoint(classifier, optimizer, epoch, mIoU, save_path, logger)
        
        log_string(f'Best mIoU so far: {best_iou:.6f}')

    save_and_plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, dirs['experiment'])

if __name__ == '__main__':
    args, config = parse_args()
    main(args, config)
