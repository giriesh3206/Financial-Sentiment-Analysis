import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add parent directory to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_config, config_path_lookup
from src.preprocessing import Vocabulary
from src.models.rnn import RNNClassifier
from src.models.lstm import LSTMClassifier

class SentimentPredictor:
    def __init__(self, model_type, models_dir=None, config_path="config.yaml"):
        """
        Loads the trained model checkpoint and vocabulary/tokenizer for inference.
        """
        self.model_type = model_type.lower()
        self.config = get_config(config_path)
        
        # Determine paths
        config_dir = os.path.dirname(os.path.abspath(config_path_lookup(config_path)))
        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(config_dir, self.config.get("models_dir", "./models")))
        
        self.labels_map = {int(k): v for k, v in self.config["labels"].items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_type in ["rnn", "lstm"]:
            # Load Vocabulary
            vocab_path = os.path.join(models_dir, f"{self.model_type}_vocab.json")
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please train the model first.")
            self.vocab = Vocabulary.load(vocab_path)
            self.max_len = self.config["rnn_lstm"].get("max_len", 64)
            
            # Initialize model architecture
            vocab_size = len(self.vocab)
            embedding_dim = self.config["rnn_lstm"]["embedding_dim"]
            hidden_dim = self.config["rnn_lstm"]["hidden_dim"]
            num_layers = self.config["rnn_lstm"]["num_layers"]
            
            if self.model_type == "rnn":
                self.model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=3, num_layers=num_layers, dropout=0.0)
            else:
                self.model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=3, num_layers=num_layers, dropout=0.0)
                
            checkpoint_path = os.path.join(models_dir, f"{self.model_type}_best.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
                
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
        elif self.model_type == "finbert":
            checkpoint_path = os.path.join(models_dir, "finbert_best")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"FinBERT checkpoint directory not found at {checkpoint_path}. Please train the model first.")
            
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            self.max_len = self.config["finbert"].get("max_len", 64)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Select 'rnn', 'lstm', or 'finbert'.")

    def predict(self, text):
        """
        Runs sentiment prediction on a raw text string.
        Returns a dictionary containing:
            - predicted_label (str)
            - confidence (float)
            - probabilities (dict: label_name -> float)
            - inference_time_ms (float)
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "predicted_label": "Neutral",
                "confidence": 1.0,
                "probabilities": {"Bearish": 0.0, "Bullish": 0.0, "Neutral": 1.0},
                "inference_time_ms": 0.0
            }
            
        start_time = time.time()
        
        with torch.no_grad():
            if self.model_type in ["rnn", "lstm"]:
                # Preprocess & Numericalize
                indices = self.vocab.numericalize(text, max_len=self.max_len)
                inputs = torch.tensor([indices], dtype=torch.long).to(self.device)
                
                # Forward pass
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            elif self.model_type == "finbert":
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=self.max_len, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                
        inference_time = (time.time() - start_time) * 1000 # ms
        
        pred_idx = np.argmax(probs)
        predicted_label = self.labels_map[pred_idx]
        confidence = float(probs[pred_idx])
        
        probabilities = {self.labels_map[i]: float(probs[i]) for i in range(len(probs))}
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "inference_time_ms": inference_time
        }

if __name__ == "__main__":
    # Test script if checkpoints exist
    # Since checkpoints are not trained yet, it will fail gracefully
    try:
        predictor = SentimentPredictor("rnn")
        res = predictor.predict("Bullish signals for Apple stock today!")
        print(res)
    except Exception as e:
        print("Inference test failed as expected (models not trained yet):", e)
