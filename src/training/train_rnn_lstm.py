import os
import sys
import argparse
import time
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_loader import load_data, get_config, config_path_lookup
from src.preprocessing import Vocabulary, clean_text
from src.models.rnn import RNNClassifier
from src.models.lstm import LSTMClassifier
from src.evaluation.eval_utils import (
    compute_metrics,
    plot_confusion_matrix,
    save_classification_report,
    save_error_analysis,
)

class FinancialTweetsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Numericalize
        indices = self.vocab.numericalize(text, max_len=self.max_len)
        
        return {
            "input_ids": torch.tensor(indices, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

def log_experiment(model_name, config, train_time, metrics, filepath="./reports/experiment_log.csv"):
    """Logs the model run metrics and parameters to a central CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    
    # We serialize relevant config parameters for auditability
    row = {
        "model": model_name,
        "embedding_dim": config["rnn_lstm"]["embedding_dim"],
        "hidden_dim": config["rnn_lstm"]["hidden_dim"],
        "num_layers": config["rnn_lstm"]["num_layers"],
        "dropout": config["rnn_lstm"]["dropout"],
        "learning_rate": config["rnn_lstm"]["learning_rate"],
        "batch_size": config["rnn_lstm"]["batch_size"],
        "train_time_sec": f"{train_time:.2f}",
        "val_accuracy": f"{metrics['accuracy']:.4f}",
        "val_macro_f1": f"{metrics['macro_f1']:.4f}",
        "val_precision": f"{metrics['precision']:.4f}",
        "val_recall": f"{metrics['recall']:.4f}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    headers = list(row.keys())
    
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged experiment details to {filepath}")

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics["accuracy"]

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return val_loss, metrics, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Train RNN or LSTM classifier.")
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "lstm"], help="Model type to train: 'rnn' or 'lstm'")
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Set seed for reproducibility
    seed = config["rnn_lstm"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load raw dataset
    train_df, val_df = load_data()
    
    # Build vocabulary from training data only to avoid leakage
    vocab = Vocabulary(
        max_size=10000, 
        min_freq=2
    )
    vocab.build_vocab(train_df["text"])
    
    # Save vocabulary
    vocab_path = os.path.join(config["models_dir"], f"{args.model}_vocab.json")
    vocab.save(vocab_path)
    
    # Create datasets and dataloaders
    max_len = config["rnn_lstm"].get("max_len", 64)
    train_dataset = FinancialTweetsDataset(train_df["text"], train_df["label"], vocab, max_len)
    val_dataset = FinancialTweetsDataset(val_df["text"], val_df["label"], vocab, max_len)
    
    batch_size = config["rnn_lstm"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    vocab_size = len(vocab)
    embedding_dim = config["rnn_lstm"]["embedding_dim"]
    hidden_dim = config["rnn_lstm"]["hidden_dim"]
    num_layers = config["rnn_lstm"]["num_layers"]
    dropout = config["rnn_lstm"]["dropout"]
    
    if args.model == "rnn":
        model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=3, num_layers=num_layers, dropout=dropout)
    else:
        model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=3, num_layers=num_layers, dropout=dropout)
        
    model.to(device)
    print(model)
    
    # Setup optimizer and loss
    learning_rate = config["rnn_lstm"]["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training variables
    epochs = config["rnn_lstm"]["epochs"]
    patience = config["rnn_lstm"].get("early_stopping_patience", 3)
    
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = config["models_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_best.pt")
    
    print(f"Starting training for {args.model.upper()}...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | Duration: {epoch_duration:.1f}s")
              
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    total_train_time = time.time() - start_time
    print(f"Training completed in {total_train_time:.2f} seconds.")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(checkpoint_path))
    val_loss, final_metrics, preds, targets = evaluate(model, val_loader, criterion, device)
    
    print("\n--- Final Validation Metrics ---")
    for k, v in final_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    # Save class-wise metrics and representative validation errors.
    class_names = [
        config["labels"][i] for i in sorted(config["labels"].keys())
    ]
    report_path = save_classification_report(
        targets,
        preds,
        class_names,
        args.model,
        config["reports_dir"],
    )
    error_path, error_count = save_error_analysis(
        val_df["text"],
        targets,
        preds,
        class_names,
        args.model,
        config["reports_dir"],
    )
    print(f"Saved class-wise report to {report_path}")
    print(f"Saved {error_count} misclassified examples to {error_path}")

    # Plot & Save Confusion Matrix
    plot_confusion_matrix(
        targets,
        preds,
        class_names,
        args.model.upper(),
        config["figures_dir"],
    )

    print("\n--- Class-wise Validation Metrics ---")
    print(pd.read_csv(report_path).to_string(index=False))

    # Log to Experiment Log
    log_experiment(
        args.model.upper(),
        config,
        total_train_time,
        final_metrics,
        os.path.join(config["reports_dir"], "experiment_log.csv"),
    )

if __name__ == "__main__":
    main()
