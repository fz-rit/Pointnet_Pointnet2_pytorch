from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def calc_metrics(true_flat: np.ndarray, 
                 pred_flat: np.ndarray, 
                 num_classes: int) -> Tuple[np.ndarray, float, float, float, float, float]:
    """
    Calculate evaluation metrics for semantic segmentation.

    Args:
    true_flat (numpy.ndarray): Flattened ground truth mask (1D array).
    pred_flat (numpy.ndarray): Flattened predicted mask (1D array).
    num_classes (int): Number of classes in the dataset.

    Returns:
    cm (numpy.ndarray): Confusion matrix.
    overall_accuracy (float): Overall accuracy.
    mAcc (float): Mean class accuracy.
    mIoU (float): Mean intersection over union.
    FWIoU (float): Frequency weighted intersection over union.
    dice_coefficient (float): Dice coefficient.
    """

    # Compute confusion matrix
    conf_mtx = confusion_matrix(true_flat, pred_flat, labels=np.arange(num_classes))

    intersection = np.diag(conf_mtx)
    # Overall Accuracy
    overall_accuracy = intersection.sum() / conf_mtx.sum()

    # Mean class Accuracy
    class_accuracy = intersection / conf_mtx.sum(axis=1)
    mAcc = np.nanmean(class_accuracy)

    # Intersection over Union (IoU) for each class
    union = conf_mtx.sum(axis=1) + conf_mtx.sum(axis=0) - np.diag(conf_mtx)
    IoUs = intersection / union 
    mIoU = np.nanmean(IoUs)

    # Frequency Weighted IoU
    freq = conf_mtx.sum(axis=1) / conf_mtx.sum()
    FWIoU = (freq * IoUs).sum()

    # Dice Coefficient for each class
    dice = 2 * intersection / (conf_mtx.sum(axis=1) + conf_mtx.sum(axis=0))
    dice_coefficient = np.nanmean(dice)

    return conf_mtx, overall_accuracy, mAcc, mIoU, FWIoU, dice_coefficient, IoUs

def save_and_plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Save and plot training/validation loss and accuracy."""
    
    # Convert any tensors to CPU/numpy for plotting
    def to_numpy(data):
        if hasattr(data, 'cpu'):
            return [float(x.cpu()) if hasattr(x, 'cpu') else float(x) for x in data]
        return [float(x) for x in data]
    
    train_losses = to_numpy(train_losses)
    val_losses = to_numpy(val_losses)
    train_accs = to_numpy(train_accs)
    val_accs = to_numpy(val_accs)
    
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_accuracy_plot.png')
    plt.close()

    print(f"Loss_accuracy_plot saved to {save_dir / 'loss_accuracy_plot.png'}")