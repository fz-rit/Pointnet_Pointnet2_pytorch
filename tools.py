from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Tuple

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