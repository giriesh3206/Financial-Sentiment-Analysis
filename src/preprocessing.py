import re
import json
import os
from collections import Counter

def clean_text(text):
    """
    Cleans financial text by:
    1. Lowercasing (except tickers if desired, but we lowercase for uniformity).
    2. Replacing URLs with [URL].
    3. Replacing User Mentions with [MENTION].
    4. Removing extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    
    # Replace user mentions
    text = re.sub(r"@\S+", "[MENTION]", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def tokenize(text):
    """
    Tokenizes text by splitting on words, numbers, punctuation,
    while preserving tickers starting with $ (e.g. $AAPL) and special tokens.
    """
    # Lowercase text and extract tokens
    tokens = re.findall(r"\[url\]|\[mention\]|\$[a-zA-Z]+|[a-zA-Z0-9]+|[^\w\s]", text.lower())
    return tokens

class Vocabulary:
    def __init__(self, max_size=10000, min_freq=2, pad_token="<pad>", unk_token="<unk>"):
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        self.pad_idx = 0
        self.unk_idx = 1
        
        self.token_to_idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.idx_to_token = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}
        
    def __len__(self):
        return len(self.token_to_idx)
        
    def build_vocab(self, texts):
        """Build vocabulary from training corpus only (prevents data leakage)."""
        counter = Counter()
        for text in texts:
            cleaned = clean_text(text)
            tokens = tokenize(cleaned)
            counter.update(tokens)
            
        # Filter by min frequency
        filtered_tokens = [token for token, freq in counter.items() if freq >= self.min_freq]
        
        # Sort by frequency
        sorted_tokens = sorted(filtered_tokens, key=lambda t: counter[t], reverse=True)
        
        # Truncate to max size (subtract 2 for pad and unk tokens)
        limit = self.max_size - 2
        for token in sorted_tokens[:limit]:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            
        print(f"Vocabulary built. Size: {len(self.token_to_idx)} (min_freq={self.min_freq})")
        
    def numericalize(self, text, max_len=64):
        """Converts raw text to list of vocabulary indices with padding and truncation."""
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        
        # Convert tokens to indices
        indices = [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
        
        # Truncate if longer than max_len
        if len(indices) > max_len:
            indices = indices[:max_len]
        # Pad if shorter than max_len
        else:
            indices = indices + [self.pad_idx] * (max_len - len(indices))
            
        return indices

    def save(self, filepath):
        """Saves vocabulary as JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({
                "token_to_idx": self.token_to_idx,
                "max_size": self.max_size,
                "min_freq": self.min_freq
            }, f, indent=4)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads vocabulary from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        vocab = cls(max_size=data["max_size"], min_freq=data["min_freq"])
        vocab.token_to_idx = data["token_to_idx"]
        vocab.idx_to_token = {int(v): k for k, v in vocab.token_to_idx.items()}
        return vocab

if __name__ == "__main__":
    # Test cleaning and tokenization
    sample_text = "Check out $AAPL and $TSLA! High gains today at http://example.com @user."
    print("Original:", sample_text)
    cleaned = clean_text(sample_text)
    print("Cleaned: ", cleaned)
    tokens = tokenize(cleaned)
    print("Tokens:  ", tokens)
    
    # Test vocabulary
    vocab = Vocabulary(max_size=10, min_freq=1)
    vocab.build_vocab([sample_text])
    numericalized = vocab.numericalize("Check out $AAPL and $TSLA", max_len=10)
    print("Numericalized:", numericalized)
