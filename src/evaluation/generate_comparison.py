import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_loader import get_config

def main():
    try:
        config = get_config()
    except Exception:
        config = {
            "reports_dir": "./reports",
            "figures_dir": "./figures"
        }
        
    reports_dir = config.get("reports_dir", "./reports")
    figures_dir = config.get("figures_dir", "./figures")
    
    log_path = os.path.join(reports_dir, "experiment_log.csv")
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}. Train the models first!")
        sys.exit(1)
        
    df = pd.read_csv(log_path)
    print("--- Experiment Benchmarks ---")
    print(df)
    
    # Check if there are models logged
    if len(df) == 0:
        print("Warning: Log file is empty.")
        sys.exit(0)
        
    # Set seaborn style for clean, professional aesthetics
    sns.set_theme(style="darkgrid")
    
    # Visualize Accuracy and F1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Validation Accuracy
    sns.barplot(
        data=df, 
        x="model", 
        y="val_accuracy", 
        ax=axes[0], 
        palette="Blues_d",
        hue="model",
        legend=False
    )
    axes[0].set_title("Validation Accuracy Comparison", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("Model Architecture", fontsize=12, labelpad=10)
    axes[0].set_ylabel("Accuracy", fontsize=12, labelpad=10)
    axes[0].set_ylim(0, 1.0)
    
    # Add values on top of bars
    for p in axes[0].patches:
        height = p.get_height()
        if not pd.isna(height):
            axes[0].annotate(
                f"{height*100:.2f}%",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=11, fontweight="semibold"
            )
            
    # 2. Validation Macro F1
    sns.barplot(
        data=df, 
        x="model", 
        y="val_macro_f1", 
        ax=axes[1], 
        palette="Greens_d",
        hue="model",
        legend=False
    )
    axes[1].set_title("Validation Macro F1 Comparison", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_xlabel("Model Architecture", fontsize=12, labelpad=10)
    axes[1].set_ylabel("Macro F1-Score", fontsize=12, labelpad=10)
    axes[1].set_ylim(0, 1.0)
    
    # Add values on top of bars
    for p in axes[1].patches:
        height = p.get_height()
        if not pd.isna(height):
            axes[1].annotate(
                f"{height*100:.2f}%",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=11, fontweight="semibold"
            )
            
    plt.tight_layout()
    
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "model_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved successfully to {out_path}")

if __name__ == "__main__":
    main()
