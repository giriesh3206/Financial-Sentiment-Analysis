# Financial News Sentiment Classification & Benchmark Hub

A pair-programming benchmarking portfolio project comparing **RNN, LSTM, and FinBERT** architectures for a 3-class financial sentiment classification task (Bearish vs. Bullish vs. Neutral). Deployed via a modern **Streamlit** dashboard.

## 🚀 Repository Structure
```
financial-news-sentiment/
├── data/                    # Local CSV cache of dataset splits
├── notebooks/
│   ├── phase1_eda.ipynb     # Interactive EDA notebook
│   └── financial_sentiment_benchmarking.ipynb  # Master Colab/Local benchmarking notebook
├── src/
│   ├── models/
│   │   ├── rnn.py           # PyTorch RNNClassifier
│   │   └── lstm.py          # PyTorch LSTMClassifier
│   ├── training/
│   │   ├── train_rnn_lstm.py # Generic baseline training script
│   │   └── train_finbert.py  # Trainer script for ProsusAI/finbert
│   ├── evaluation/
│   │   └── eval_utils.py    # Metric calculators & confusion matrix plotters
│   ├── data_loader.py       # Hugging Face downloader & local cacher
│   ├── preprocessing.py     # Cleaners, tokenizers, Vocabulary class
│   └── phase1_eda.py        # Standalone EDA CLI script
├── models/                  # Saved weights (*_best.pt) and vocab files
├── reports/
│   └── experiment_log.csv   # Unified CSV experiment metrics logger
├── figures/                 # Confusion matrices and distribution plots
├── app.py                   # Streamlit inference dashboard
├── config.yaml              # Hyperparameters and folder paths
├── requirements.txt         # Pinned python packages
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 📊 Benchmark Results (Real Measured Metrics)

The models were evaluated on the **Twitter Financial News Sentiment** dataset (Hugging Face: `zeroshot/twitter-financial-news-sentiment`).

### 1. Dataset Statistics (Phase 1 EDA)
- **Train split size:** 9,543 samples (79.98%)
- **Validation split size:** 2,388 samples (20.02%)
- **Train/Val text overlap:** 0.00% (No data leakage)
- **Class Imbalance:**
  - `LABEL_0` (Bearish): 15.11%
  - `LABEL_1` (Bullish): 20.15%
  - `LABEL_2` (Neutral): **64.74%**
  
*Note: Due to the 65% Neutral class dominance, Accuracy is naturally high. **Macro F1** is utilized as the primary metric for comparative benchmarks.*

### 2. Model Performance Benchmarks
RNN and LSTM baseline models were trained in this workspace on CPU. The FinBERT model was fine-tuned in this workspace on an **NVIDIA GeForce RTX 4050 Laptop GPU**:

| Model Architecture | Epochs | Train Loss | Val Loss | Val Accuracy | Val Macro F1 | Train Time |
|---|---|---|---|---|---|---|
| **Simple RNN** | 13 | 0.4670 | 0.5608 | 78.48% | 66.54% | 213.51s |
| **LSTM** | 11 | 0.4039 | 0.5190 | 81.24% | 72.14% | 179.48s |
| **FinBERT** | 3 | **0.3165** | **0.4728** | **87.65%** | **83.67%** | **372.63s** |

---

## 🛠️ Getting Started & Local Usage

### 1. Set Up Environment & Install Requirements
Create a virtual environment and install the requirements:
```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 2. Run EDA CLI
Run the exploratory analysis CLI:
```bash
python src/phase1_eda.py
```

### 3. Train Models
Train the baseline models locally:
```bash
python src/training/train_rnn_lstm.py --model rnn
python src/training/train_rnn_lstm.py --model lstm
```

### 4. Run Streamlit Dashboard
Launch the dashboard for real-time model inference and metric visualization:
```bash
streamlit run app.py
```

---

## ☁️ Running on Google Colab (GPU training)
To train the GPU-heavy **FinBERT** model or run the code on Google Colab:
1. Upload this folder to your Google Drive.
2. Open [notebooks/financial_sentiment_benchmarking.ipynb](file:///notebooks/financial_sentiment_benchmarking.ipynb) in Colab.
3. Enable GPU acceleration (`Runtime` -> `Change runtime type` -> select `T4 GPU`).
4. Execute the Colab setup block to mount Drive and install requirements.
5. Run the FinBERT training block; the script will fine-tune the model, save weights directly to your Google Drive, and log benchmarks.

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](file:///LICENSE) file for details.
