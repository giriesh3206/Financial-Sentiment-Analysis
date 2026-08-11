import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=3, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Simple RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, embedding_dim]
        
        # RNN outputs: 
        # out: [batch_size, seq_len, hidden_dim]
        # h_n: [num_layers, batch_size, hidden_dim]
        out, _ = self.rnn(embedded)
        
        # Mean pooling over the sequence length dimension (dim=1) to aggregate features
        pooled = torch.mean(out, dim=1)  # [batch_size, hidden_dim]
        
        logits = self.fc(self.dropout(pooled))  # [batch_size, num_classes]
        return logits
