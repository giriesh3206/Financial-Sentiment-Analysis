import os
import yaml
import pandas as pd
from datasets import load_dataset

EXPECTED_TRAIN_SIZE = 9938
EXPECTED_VALIDATION_SIZE = 2486


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


def _has_required_split_sizes(train_df, val_df):
    """Check that local data matches the assignment's official split sizes."""
    return (
        len(train_df) == EXPECTED_TRAIN_SIZE
        and len(val_df) == EXPECTED_VALIDATION_SIZE
    )


def load_data():
    """
    Load the required training and validation datasets.

    Local caches are used only when they match the assignment's official
    split sizes. Otherwise the official Hugging Face splits are downloaded.
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

    train_df = None
    val_df = None

    if os.path.exists(custom_train_path) and os.path.exists(custom_val_path):
        print(f"Loading custom Twitter dataset from {abs_data_dir}...")
        candidate_train = pd.read_csv(custom_train_path)
        candidate_val = pd.read_excel(custom_val_path)
        if _has_required_split_sizes(candidate_train, candidate_val):
            train_df, val_df = candidate_train, candidate_val
        else:
            print(
                "Local custom split sizes do not match the assignment "
                f"({EXPECTED_TRAIN_SIZE}/{EXPECTED_VALIDATION_SIZE}). "
                "Ignoring them and loading the official Hugging Face splits."
            )

    if train_df is None or val_df is None:
        if os.path.exists(train_path) and os.path.exists(val_path):
            print(f"Loading cached dataset from {abs_data_dir}...")
            candidate_train = pd.read_csv(train_path)
            candidate_val = pd.read_csv(val_path)
            if _has_required_split_sizes(candidate_train, candidate_val):
                train_df, val_df = candidate_train, candidate_val
            else:
                print(
                    "Local cached split sizes do not match the assignment "
                    f"({EXPECTED_TRAIN_SIZE}/{EXPECTED_VALIDATION_SIZE}). "
                    "Ignoring the cache and downloading the official "
                    "Hugging Face splits."
                )

    if train_df is None or val_df is None:
        print("Downloading the official dataset from Hugging Face...")
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
        train_df = pd.DataFrame(dataset["train"])
        val_df = pd.DataFrame(dataset["validation"])

        if not _has_required_split_sizes(train_df, val_df):
            raise ValueError(
                "The downloaded dataset split sizes do not match the assignment: "
                f"expected {EXPECTED_TRAIN_SIZE}/{EXPECTED_VALIDATION_SIZE}, "
                f"received {len(train_df)}/{len(val_df)}."
            )

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        print(f"Saved official raw splits to {abs_data_dir}")

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
