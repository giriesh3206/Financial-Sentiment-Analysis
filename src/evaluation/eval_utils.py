import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Computes classification metrics: accuracy, precision, recall, and macro F1.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall
    }

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir="./figures"):
    """
    Plots and saves a confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Use sleek styling with blue tones
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")
    return save_path
