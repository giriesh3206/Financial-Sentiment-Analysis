import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    """Compute accuracy and macro-averaged precision, recall, and F1."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
    }


def get_classification_report(y_true, y_pred, class_names):
    """Return class-wise precision, recall, F1, and support as a DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(
        [
            {
                "class": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in class_names
        ]
    )


def save_classification_report(
    y_true, y_pred, class_names, model_name, save_dir="./reports"
):
    """Save class-wise precision, recall, F1 and support to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    report_df = get_classification_report(y_true, y_pred, class_names)
    path = os.path.join(
        save_dir,
        f"classification_report_{model_name.lower().replace(' ', '_')}.csv",
    )
    report_df.to_csv(path, index=False)
    return path


def save_error_analysis(
    texts, y_true, y_pred, class_names, model_name, save_dir="./reports", max_rows=100
):
    """Save misclassified examples for qualitative error analysis."""
    os.makedirs(save_dir, exist_ok=True)
    rows = [
        {
            "text": str(text),
            "actual": class_names[int(true_id)],
            "predicted": class_names[int(pred_id)],
        }
        for text, true_id, pred_id in zip(texts, y_true, y_pred)
        if int(true_id) != int(pred_id)
    ]
    error_df = pd.DataFrame(rows).head(max_rows)
    path = os.path.join(
        save_dir,
        f"error_analysis_{model_name.lower().replace(' ', '_')}.csv",
    )
    error_df.to_csv(path, index=False)
    return path, len(rows)


def plot_confusion_matrix(
    y_true, y_pred, classes, model_name, save_dir="./figures"
):
    """Plot and save a confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")
    return save_path
