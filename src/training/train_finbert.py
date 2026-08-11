import os
import sys
import time
import argparse
import pandas as pd
from datasets import Dataset
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
)

# Add parent directory to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_loader import load_data, get_config
from src.evaluation.eval_utils import compute_metrics, plot_confusion_matrix
from src.training.train_rnn_lstm import log_experiment

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT model on financial sentiment dataset.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Limit number of training steps (useful for profiling/testing).")
    parser.add_argument("--subset_size", type=int, default=-1, help="Limit the number of training samples (useful for faster training on CPU).")
    parser.add_argument("--epochs", type=int, default=-1, help="Override training epochs.")
    args = parser.parse_args()

    # Load configuration
    config = get_config()
    
    # Set seed for reproducibility
    seed = config["finbert"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load raw dataset
    train_df, val_df = load_data()

    if args.subset_size > 0:
        print(f"Subsetting data to {args.subset_size} train and {max(1, int(args.subset_size * 0.25))} validation samples...")
        train_df = train_df.sample(n=min(args.subset_size, len(train_df)), random_state=seed).reset_index(drop=True)
        val_subset_size = min(max(1, int(args.subset_size * 0.25)), len(val_df))
        val_df = val_df.sample(n=val_subset_size, random_state=seed).reset_index(drop=True)
    
    # Load FinBERT tokenizer
    model_name = config["finbert"]["model_name"]
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models/finbert_cache")
    
    # Tokenize function
    max_len = config["finbert"].get("max_len", 64)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
        
    # Convert DataFrames to HF Dataset objects
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Define mapping of labels for FinBERT
    # In dataset: 0 = Bearish, 1 = Bullish, 2 = Neutral
    id2label = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    label2id = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
    
    # Load model
    print(f"Loading model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # Safely override classification head mappings
        cache_dir="./models/finbert_cache"
    )
    
    # Define metric computation function for Trainer
    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds)
        
    # Define Training arguments
    # Save directory for checkpoints (can be set to Drive path if running on Colab)
    models_dir = config["models_dir"]
    output_dir = os.path.join(models_dir, "finbert_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # T4 GPU supports fp16. If cuda is available, use fp16 for faster training.
    use_fp16 = torch.cuda.is_available()
    
    num_epochs = args.epochs if args.epochs > 0 else config["finbert"]["epochs"]
    
    training_args_dict = {
        "output_dir": output_dir,
        "eval_strategy": "epoch" if args.max_steps <= 0 else "no",
        "save_strategy": "epoch" if args.max_steps <= 0 else "no",
        "learning_rate": float(config["finbert"]["learning_rate"]),
        "per_device_train_batch_size": config["finbert"]["batch_size"],
        "per_device_eval_batch_size": config["finbert"]["batch_size"],
        "num_train_epochs": num_epochs,
        "weight_decay": config["finbert"]["weight_decay"],
        "load_best_model_at_end": True if args.max_steps <= 0 else False,
        "metric_for_best_model": "eval_macro_f1" if args.max_steps <= 0 else None,
        "greater_is_better": True if args.max_steps <= 0 else None,
        "fp16": use_fp16,
        "logging_steps": 50 if args.max_steps <= 0 else 1,
        "seed": seed,
        "report_to": "none"  # Suppress wandb/mlflow logging
    }
    
    if args.max_steps > 0:
        training_args_dict["max_steps"] = args.max_steps
        
    training_args = TrainingArguments(**training_args_dict)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] if args.max_steps <= 0 else None
    )
    
    # Train
    print("Starting FinBERT fine-tuning...")
    start_time = time.time()
    trainer.train()
    total_train_time = time.time() - start_time
    print(f"FinBERT training completed in {total_train_time:.2f} seconds.")
    
    if args.max_steps > 0:
        print(f"FinBERT profiling run finished. Total training time for {args.max_steps} steps: {total_train_time:.2f} seconds.")
        print(f"Average time per step: {total_train_time / args.max_steps:.2f} seconds.")
        return
        
    # Evaluate
    print("Evaluating FinBERT on validation set...")
    eval_results = trainer.evaluate()
    
    # Get final predictions for confusion matrix
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    targets = preds_output.label_ids
    
    final_metrics = {
        "accuracy": eval_results["eval_accuracy"],
        "macro_f1": eval_results["eval_macro_f1"],
        "precision": eval_results["eval_precision"],
        "recall": eval_results["eval_recall"]
    }
    
    print("\n--- FinBERT Final Validation Metrics ---")
    for k, v in final_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    # Save the best model locally
    best_model_path = os.path.join(models_dir, "finbert_best")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Saved best FinBERT model to {best_model_path}")
    
    # Plot & Save Confusion Matrix
    class_names = [config["labels"][i] for i in sorted(config["labels"].keys())]
    plot_confusion_matrix(targets, preds, class_names, "FINBERT", config["figures_dir"])
    
    # Log to Experiment Log
    # Map finbert config to fit generic log_experiment signature
    rnn_lstm_dummy_config = {
        "rnn_lstm": {
            "embedding_dim": "N/A (Transformer)",
            "hidden_dim": "768 (FinBERT)",
            "num_layers": "12",
            "dropout": "0.1",
            "learning_rate": config["finbert"]["learning_rate"],
            "batch_size": config["finbert"]["batch_size"]
        }
    }
    log_experiment("FINBERT", rnn_lstm_dummy_config, total_train_time, final_metrics, os.path.join(config["reports_dir"], "experiment_log.csv"))

if __name__ == "__main__":
    main()
