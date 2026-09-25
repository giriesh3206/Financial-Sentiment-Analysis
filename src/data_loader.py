import os
import yaml
import pandas as pd
from datasets import load_dataset


def get_config(config_path="config.yaml"):
    """Load YAML configuration from project root."""
    path = config_path
    for _ in range(3):
        if os.path.exists(path):
            break
        path = os.path.join("..", path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Configuration file config.yaml not found "
            "(searched up to 3 parent directories)."
        )

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _validate_schema(train_df, val_df):
    """Validate that both splits contain the expected text/label columns."""
    required_columns = {"text", "label"}

    for name, df in (("train", train_df), ("validation", val_df)):
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} split is missing required columns: "
                f"{sorted(missing)}."
            )

        if df["text"].isna().any() or df["label"].isna().any():
            raise ValueError(f"{name} split contains null text or label values.")

    train_labels = set(train_df["label"].unique())
    val_labels = set(val_df["label"].unique())
    allowed_labels = {0, 1, 2, "0", "1", "2"}

    if not train_labels.issubset(allowed_labels) or not val_labels.issubset(
        allowed_labels
    ):
        raise ValueError(
            "Unexpected labels found. Expected the three labels "
            "LABEL_0/LABEL_1/LABEL_2 or encoded values 0/1/2."
        )


def _report_split_sizes(train_df, val_df, source):
    """Print the actual split sizes without modifying the supplied data."""
    print(
        f"Using {source} train/validation splits: "
        f"{len(train_df)}/{len(val_df)} samples."
    )


def load_data():
    """
    Load the training and validation datasets.

    The project uses only the supplied Hugging Face train and validation
    splits. Local copies are preferred when present. Split sizes are reported
    but are not changed, padded, duplicated, or synthetically expanded.
    """
    config = get_config()
    data_dir = config.get("data_dir", "./data")

    config_dir = os.path.dirname(os.path.abspath(config_path_lookup()))
    abs_data_dir = os.path.abspath(os.path.join(config_dir, data_dir))
    os.makedirs(abs_data_dir, exist_ok=True)

    custom_train_path = os.path.join(abs_data_dir, "sent_train.csv")
    custom_val_path = os.path.join(abs_data_dir, "sent_valid.xlsx")
    train_path = os.path.join(abs_data_dir, "train_raw.csv")
    val_path = os.path.join(abs_data_dir, "validation_raw.csv")

    if os.path.exists(custom_train_path) and os.path.exists(custom_val_path):
        print(f"Loading custom Twitter dataset from {abs_data_dir}...")
        train_df = pd.read_csv(custom_train_path)
        val_df = pd.read_excel(custom_val_path)
        _validate_schema(train_df, val_df)
        _report_split_sizes(train_df, val_df, "local custom")
        return train_df, val_df

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Loading cached dataset from {abs_data_dir}...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        _validate_schema(train_df, val_df)
        _report_split_sizes(train_df, val_df, "cached official")
        return train_df, val_df

    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])

    _validate_schema(train_df, val_df)
    _report_split_sizes(train_df, val_df, "Hugging Face")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"Saved raw splits to {abs_data_dir}")

    return train_df, val_df


def config_path_lookup(config_name="config.yaml"):
    """Locate config.yaml relative to the current working directory."""
    path = config_name
    for _ in range(3):
        if os.path.exists(path):
            return path
        path = os.path.join("..", path)
    raise FileNotFoundError(f"Configuration file {config_name} not found.")


if __name__ == "__main__":
    print("Testing data loader...")
    try:
        train, val = load_data()
        print(f"Train split size: {len(train)}")
        print(f"Validation split size: {len(val)}")
        print("Sample training data:")
        print(train.head(3))
    except Exception as e:
        print("Error during loading:", e)
