import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data, get_config

def run_eda():
    print("==================================================")
    print("PHASE 1: Dataset & Exploratory Data Analysis (EDA)")
    print("==================================================\n")
    
    # Load dataset
    train_df, val_df = load_data()
    config = get_config()
    labels_map = {int(k): v for k, v in config["labels"].items()}
    figures_dir = config.get("figures_dir", "./figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Verify split sizes
    train_size = len(train_df)
    val_size = len(val_df)
    total_size = train_size + val_size
    print(f"--- Split Sizes ---")
    print(f"Train samples:      {train_size} ({train_size/total_size*100:.2f}%)")
    print(f"Validation samples: {val_size} ({val_size/total_size*100:.2f}%)")
    print(f"Total dataset:      {total_size}\n")
    
    # 2. Check class distribution
    print("--- Class Distributions ---")
    for df_name, df in [("Train", train_df), ("Validation", val_df)]:
        print(f"\n{df_name} Split:")
        dist = df["label"].value_counts().sort_index()
        for label_id, count in dist.items():
            label_name = labels_map.get(label_id, f"Unknown ({label_id})")
            pct = count / len(df) * 100
            print(f"  LABEL_{label_id} ({label_name:<7}): {count:>5} samples ({pct:.2f}%)")
            
    # Save distribution plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    train_counts = train_df["label"].map(labels_map).value_counts()
    sns.barplot(x=train_counts.index, y=train_counts.values, palette="Blues_d")
    plt.title("Train Label Distribution")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    val_counts = val_df["label"].map(labels_map).value_counts()
    sns.barplot(x=val_counts.index, y=val_counts.values, palette="Greens_d")
    plt.title("Validation Label Distribution")
    plt.ylabel("Count")
    
    plt.tight_layout()
    dist_plot_path = os.path.join(figures_dir, "class_distribution.png")
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved class distribution plot to {dist_plot_path}\n")
    
    # 3. Tweet length statistics (word count and character count)
    print("--- Tweet Length Stats (Word Count) ---")
    for df_name, df in [("Train", train_df), ("Validation", val_df)]:
        word_counts = df["text"].apply(lambda t: len(str(t).split()))
        char_counts = df["text"].apply(lambda t: len(str(t)))
        
        print(f"\n{df_name} Word Counts:")
        print(f"  Min words:    {word_counts.min()}")
        print(f"  Max words:    {word_counts.max()}")
        print(f"  Mean words:   {word_counts.mean():.2f}")
        print(f"  Median words: {word_counts.median():.1f}")
        print(f"  Std dev:      {word_counts.std():.2f}")
        
    # Save length distribution plot
    plt.figure(figsize=(10, 5))
    train_word_counts = train_df["text"].apply(lambda t: len(str(t).split()))
    val_word_counts = val_df["text"].apply(lambda t: len(str(t).split()))
    
    sns.histplot(train_word_counts, color="blue", label="Train", kde=True, bins=30, alpha=0.5)
    sns.histplot(val_word_counts, color="green", label="Validation", kde=True, bins=30, alpha=0.5)
    plt.xlabel("Word Count")
    plt.ylabel("Density")
    plt.title("Tweet Word Length Distribution")
    plt.legend()
    
    len_plot_path = os.path.join(figures_dir, "word_length_distribution.png")
    plt.savefig(len_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved word length distribution plot to {len_plot_path}\n")
    
    # 4. Missing / duplicate rows
    print("--- Missing & Duplicate Rows ---")
    for df_name, df in [("Train", train_df), ("Validation", val_df)]:
        null_texts = df["text"].isnull().sum()
        null_labels = df["label"].isnull().sum()
        duplicates = df.duplicated(subset=["text"]).sum()
        print(f"\n{df_name} Data Quality:")
        print(f"  Null texts:     {null_texts}")
        print(f"  Null labels:    {null_labels}")
        print(f"  Duplicate texts: {duplicates}")
        
    # 5. Train / Validation Overlap Check
    print("\n--- Train/Validation Overlap Check ---")
    train_texts_set = set(train_df["text"].str.strip().str.lower())
    val_texts = val_df["text"].str.strip().str.lower()
    
    overlap_count = 0
    overlap_samples = []
    for text in val_texts:
        if text in train_texts_set:
            overlap_count += 1
            if len(overlap_samples) < 3:
                overlap_samples.append(text)
                
    overlap_pct = (overlap_count / len(val_df)) * 100
    print(f"Overlap size: {overlap_count} validation tweets exist in train set ({overlap_pct:.2f}% overlap)")
    if overlap_count > 0:
        print("Overlap samples:")
        for s in overlap_samples:
            print(f"  - \"{s[:80]}...\"")
    print("")
            
    # 6. Sample Tweets per class
    print("--- Sample Tweets per Class ---")
    for label_id in sorted(labels_map.keys()):
        label_name = labels_map[label_id]
        print(f"\nSentiment: {label_name.upper()} (LABEL_{label_id})")
        samples = train_df[train_df["label"] == label_id]["text"].head(3).tolist()
        for idx, sample in enumerate(samples, 1):
            print(f"  {idx}. \"{sample}\"")
            
    print("\n==================================================")
    print("Phase 1 EDA analysis complete.")
    print("==================================================")

if __name__ == "__main__":
    run_eda()
