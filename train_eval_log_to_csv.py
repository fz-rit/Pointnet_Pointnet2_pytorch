import re
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def extract_metrics_from_log_txt(log_txt_path, out_csv_path):
    """
    Extracts metrics from a log file and returns them as a DataFrame.
    
    Args:
        log_txt_path (str): Path to the log file.
    
    Returns:
        pd.DataFrame: DataFrame containing the extracted metrics.
    """
    
    # Load the log file
    
    with open(log_txt_path, "r") as file:
        log_lines = file.readlines()

    # Initialize list to hold parsed data
    epochs_data = []

    # Initialize current epoch dictionary
    current_epoch = {}

    # Regex patterns
    epoch_pattern = re.compile(r"\*\*\*\* Epoch (\d+)")
    train_loss_pattern = re.compile(r"Training mean loss: ([\d.]+)")
    train_acc_pattern = re.compile(r"Training accuracy: ([\d.]+)")
    val_loss_pattern = re.compile(r"eval mean loss: ([\d.]+)")
    val_iou_pattern = re.compile(r"eval point avg class IoU: ([\d.]+)")
    val_acc_pattern = re.compile(r"eval point accuracy: ([\d.]+)")
    val_avg_class_acc_pattern = re.compile(r"eval point avg class acc: ([\d.]+)")
    class_iou_pattern = re.compile(r"class (\w+)\s+weight: [\d.]+, IoU: ([\d.]+)")

    # Loop through lines to extract data
    for line in log_lines:
        if match := epoch_pattern.search(line):
            if current_epoch:
                epochs_data.append(current_epoch)
            current_epoch = {"Epoch": int(match.group(1))}
        elif match := train_loss_pattern.search(line):
            current_epoch["Training mean loss"] = float(match.group(1))
        elif match := train_acc_pattern.search(line):
            current_epoch["Training accuracy"] = float(match.group(1))
        elif match := val_loss_pattern.search(line):
            current_epoch["Validation mean loss"] = float(match.group(1))
        elif match := val_iou_pattern.search(line):
            current_epoch["Validation point avg class IoU"] = float(match.group(1))
        elif match := val_acc_pattern.search(line):
            current_epoch["Validation point accuracy"] = float(match.group(1))
        elif match := val_avg_class_acc_pattern.search(line):
            current_epoch["Validation point avg class acc"] = float(match.group(1))
        elif match := class_iou_pattern.search(line):
            class_name = match.group(1)
            current_epoch[f"IoU {class_name}"] = float(match.group(2))

    # Append the final epoch
    if current_epoch:
        epochs_data.append(current_epoch)

    # Convert to DataFrame
    df = pd.DataFrame(epochs_data)

    # Save to CSV
    csv_path = "pointnet2_sem_seg_metrics.csv"  # Output file
    df.to_csv(csv_path, index=False)

    print(f"Saved metrics to {csv_path}")
    return df





def plot_train_metrics(df):
    # Plot 1: Losses over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Training mean loss'], '-o', label='Training Loss')
    plt.plot(df['Epoch'], df['Validation mean loss'], '-x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 2: Accuracy and IoU over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Training accuracy'], '-o', label='Training Accuracy')
    plt.plot(df['Epoch'], df['Validation point accuracy'], '-o', label='Validation Accuracy')
    plt.plot(df['Epoch'], df['Validation point avg class IoU'], '-o', label='Validation Avg Class IoU')
    plt.plot(df['Epoch'], df['Validation point avg class acc'], '-o', label='Validation Avg Class Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Accuracy and IoU over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 3: IoU for each class
    class_iou_columns = [col for col in df.columns if col.startswith('IoU')]
    plt.figure(figsize=(10, 6))
    for col in class_iou_columns:
        plt.plot(df['Epoch'], df[col], '-o', label=col)
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU for Each Class over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, title):

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()



def make_a_table_from_eval_log(metrics_summary):
    """
    Create a table from the evaluation log metrics summary.
    Args:
        metrics_summary (list): List of dictionaries containing metrics for each file.

    """

    # Convert to DataFrame and display
    metrics_df = pd.DataFrame(metrics_summary)
    out_csv_path = Path("./results_metrics.csv")
    metrics_df.to_csv(out_csv_path, index=False)
    print(f"Saved metrics summary to {out_csv_path}")



def plot_eval_metrics(eval_log_path, out_csv_path):

    with open(eval_log_path, "r") as file:
        eval_lines = file.read()

    results = eval_lines.split("------- Evaluation Results --------")
    confusion_data = {}
    metrics_summary = []
    class_names = ['Ground', 'Stem', 'Canopy', 'Roots', 'Objects']

    for result in results[1:]:
        filename_match = re.search(r"Save the prediction to .*/([^/]+)_pred\.csv", result)
        if not filename_match:
            continue
        filename = filename_match.group(1)

        cm_match = re.findall(r"\[\[?([^]]+)\]?", result)
        if not cm_match or len(cm_match) < 5:
            continue
        cm_cleaned = []
        for row in cm_match[:5]:
            cleaned_row = row.replace('[', '').replace(']', '').strip()
            cm_cleaned.append([int(num) for num in cleaned_row.split()])
        confusion_matrix = np.array(cm_cleaned)
        confusion_data[filename] = confusion_matrix

        def extract_metric(pattern):
            match = re.search(pattern, result)
            return float(match.group(1)) if match else None

        metrics = {
            "File": filename,
            "Overall Accuracy": extract_metric(r"Overall Accuracy:\s+([0-9.]+)"),
            "Mean Class Accuracy": extract_metric(r"Mean Class Accuracy:\s+([0-9.]+)"),
            "Mean IoU": extract_metric(r"Mean IoU:\s+([0-9.]+)"),
            "Freq Weighted IoU": extract_metric(r"Frequency Weighted IoU:\s+([0-9.]+)"),
            "Dice Coefficient": extract_metric(r"Dice Coefficient:\s+([0-9.]+)"),
        }

        for class_name in class_names:
            metrics[f"IoU {class_name}"] = extract_metric(rf"{class_name}:\s+([0-9.]+)")

        metrics_summary.append(metrics)

    # Plot confusion matrices
    for filename, cm in confusion_data.items():
        plot_confusion_matrix(cm, class_names, title=f"Confusion Matrix\n{filename}")

    # Save metrics summary to CSV
    eval_df = pd.DataFrame(metrics_summary)
    eval_df.to_csv(out_csv_path, index=False)



def main():
    # Example usage
    log_path = Path("/home/fzhcis/mylab/Pointnet_Pointnet2_pytorch/log/sem_seg/2025-04-11_11-41/logs/pointnet2_sem_seg.txt")
    out_train_csv_path = Path("./pointnet2_sem_seg_train_metrics.csv")
    out_eval_csv_path = Path("./pointnet2_sem_seg_eval_metrics.csv")
    metrics_df = extract_metrics_from_log_txt(log_path, out_train_csv_path)
    plot_train_metrics(metrics_df)
    eval_log_path = Path("/home/fzhcis/mylab/Pointnet_Pointnet2_pytorch/log/sem_seg/2025-04-11_11-41/eval.txt")
    plot_eval_metrics(eval_log_path, out_eval_csv_path)
    plt.show()


if __name__ == "__main__":
    main()
    
