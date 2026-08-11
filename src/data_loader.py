import os
import yaml
import pandas as pd
from datasets import load_dataset

def get_config(config_path="config.yaml"):
    """Loads YAML configuration from project root."""
    # Find config.yaml path in parents if needed (e.g. if running from src or notebooks)
    path = config_path
    for _ in range(3):
        if os.path.exists(path):
            break
        path = os.path.join("..", path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file config.yaml not found (searched up to 3 parent directories).")
        
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data():
    """
    Loads the Hugging Face dataset (zeroshot/twitter-financial-news-sentiment)
    and caches the splits as CSV files under the configured data directory.
    
    Returns:
        train_df (pd.DataFrame): Raw training dataset
        val_df (pd.DataFrame): Raw validation dataset
    """
    config = get_config()
    data_dir = config.get("data_dir", "./data")
    
    # Resolve absolute path relative to config.yaml directory
    config_dir = os.path.dirname(os.path.abspath(config_path_lookup()))
    abs_data_dir = os.path.abspath(os.path.join(config_dir, data_dir))
    
    os.makedirs(abs_data_dir, exist_ok=True)
    
    train_path = os.path.join(abs_data_dir, "train_raw.csv")
    val_path = os.path.join(abs_data_dir, "validation_raw.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Loading cached dataset from {abs_data_dir}...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    else:
        print("Downloading dataset from Hugging Face...")
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
        
        train_df = pd.DataFrame(dataset["train"])
        val_df = pd.DataFrame(dataset["validation"])
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        print(f"Saved raw splits to {abs_data_dir}")
        
    return train_df, val_df

def config_path_lookup(config_name="config.yaml"):
    """Helper to locate config.yaml relative to current working directory."""
    path = config_name
    for _ in range(3):
        if os.path.exists(path):
            return path
        path = os.path.join("..", path)
    raise FileNotFoundError(f"Configuration file {config_name} not found.")

if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")
    try:
        train, val = load_data()
        print(f"Train split size: {len(train)}")
        print(f"Validation split size: {len(val)}")
        print("Sample training data:")
        print(train.head(3))
    except Exception as e:
        print("Error during loading:", e)
