# Financial News Sentiment Classification & Benchmark Hub

A 3-class financial sentiment classification project comparing **Simple RNN, LSTM, and FinBERT** models for Bearish, Bullish, and Neutral financial tweets. The project includes EDA, leakage-safe preprocessing, model training, quantitative evaluation, error analysis, and a Streamlit inference dashboard.

## Repository Structure

\`\`\`
financial-news-sentiment/
├── data/                    # Local cache of the official train/validation splits
├── notebooks/
│   ├── phase1_eda.ipynb
│   └── financial_sentiment_benchmarking.ipynb
├── src/
│   ├── models/
│   │   ├── rnn.py
│   │   └── lstm.py
│   ├── training/
│   │   ├── train_rnn_lstm.py
│   │   └── train_finbert.py
│   ├── evaluation/
│   │   ├── eval_utils.py
│   │   └── generate_comparison.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── inference.py
│   └── phase1_eda.py
├── models/                  # Local model checkpoints and vocabularies
├── reports/                 # Metrics, class-wise reports, and error analysis
├── figures/                 # EDA and confusion-matrix figures
├── app.py                  # Streamlit dashboard
├── config.yaml
├── requirements.txt
└── README.md
\`\`\`

## Dataset

The project uses the required Hugging Face dataset:

\`zeroshot/twitter-financial-news-sentiment\`

The loader enforces the assignment's official split sizes:

- **Training:** 9,938
- **Validation:** 2,486
- **Labels:** Bearish, Bullish, Neutral
- **Only the provided train and validation splits are used.**

If an old local cache has different split sizes, the loader ignores it and downloads the official splits again. This prevents accidentally benchmarking a reduced or modified dataset.

## Pipeline

\`\`\`
Hugging Face dataset
        ↓
EDA + data-quality checks
        ↓
URL / mention / whitespace cleaning
        ↓
Ticker-aware tokenization
        ↓
Vocabulary built from TRAIN only
        ↓
Simple RNN + LSTM training
        ↓
Validation accuracy / macro F1 / class-wise metrics
        ↓
Confusion matrices + misclassification analysis

Same official train/validation splits
        ↓
FinBERT fine-tuning
        ↓
Validation metrics + confusion matrix + error analysis
        ↓
Model comparison
        ↓
Streamlit inference dashboard
\`\`\`

## Evaluation Outputs

After training, the project generates:

- Overall validation accuracy
- Macro F1
- Macro precision and recall
- Class-wise precision, recall, F1 and support
- Confusion matrices
- CSV files containing misclassified examples for qualitative error analysis
- Unified experiment log

Example generated report files:

\`\`\`
reports/
├── experiment_log.csv
├── classification_report_rnn.csv
├── classification_report_lstm.csv
├── classification_report_finbert.csv
├── error_analysis_rnn.csv
├── error_analysis_lstm.csv
└── error_analysis_finbert.csv
\`\`\`

The error-analysis CSVs contain the original text, actual class, and predicted class so the report can discuss concrete model failures rather than only reporting aggregate numbers.

## Model Comparison

The project compares:

1. **Simple RNN:** Embedding → 2-layer RNN → mean pooling → classifier
2. **LSTM:** Embedding → 2-layer LSTM → mean pooling → classifier
3. **FinBERT:** Pretrained \`ProsusAI/finbert\` fine-tuned for the three required classes

The LSTM/RNN comparison is interpreted in terms of recurrent architecture and sequence-memory behavior. FinBERT is compared quantitatively against the baselines using the same official validation split.

## Running the Project

### 1. Install dependencies

\`\`\`bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

### 2. Run EDA

\`\`\`bash
python src/phase1_eda.py
\`\`\`

### 3. Train the RNN and LSTM

\`\`\`bash
python src/training/train_rnn_lstm.py --model rnn
python src/training/train_rnn_lstm.py --model lstm
\`\`\`

### 4. Fine-tune FinBERT

For a GPU environment:

\`\`\`bash
python src/training/train_finbert.py
\`\`\`

A reduced run can be used only for development/profiling:

\`\`\`bash
python src/training/train_finbert.py --subset_size 1000
\`\`\`

Do not use a subset run for the final project benchmark.

### 5. Run Streamlit

\`\`\`bash
streamlit run app.py
\`\`\`

The dashboard provides:

- RNN/LSTM/FinBERT model selection
- financial text input
- predicted sentiment
- confidence
- class probabilities
- probability bar chart
- validation class-distribution chart
- model accuracy and macro-F1 comparison
- confusion matrices

## Required 2-Page Comparison Report

After running all three final models on the official splits, generate the comparison report from the measured experiment files. The report should contain:

- dataset and methodology
- RNN vs LSTM metrics
- RNN/LSTM interpretation
- FinBERT comparison
- class-wise precision/recall/F1
- confusion matrices
- representative error-analysis examples
- Streamlit description/screenshot

The report should be kept to **2 pages** for submission.

## Important Benchmark Note

Earlier development runs in this repository used a smaller cached split (9,543 training / 2,388 validation). The data loader has now been changed to enforce the assignment's required **9,938 / 2,486** split. Therefore, the final benchmark numbers and PDF report must be regenerated after the models are retrained on the official splits.

## Reproducibility

- Fixed random seed: 42
- RNN/LSTM vocabulary is built from training data only
- Validation data is never used to build the vocabulary
- Early stopping is enabled
- FinBERT uses validation-based model selection
- Experiment parameters and metrics are logged for comparison
