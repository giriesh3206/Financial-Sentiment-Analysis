import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add parent directory to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.data_loader import get_config
from src.inference import SentimentPredictor

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
try:
    config = get_config()
except Exception as e:
    config = {"labels": {0: "Bearish", 1: "Bullish", 2: "Neutral"}}

# Modern Styling Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px;
        padding: 10px 20px;
        color: #f8fafc;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        text-align: center;
    }
    .metric-title {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val-bullish {
        font-size: 2.25rem;
        color: #10b981;
        font-weight: 700;
    }
    .metric-val-bearish {
        font-size: 2.25rem;
        color: #ef4444;
        font-weight: 700;
    }
    .metric-val-neutral {
        font-size: 2.25rem;
        color: #f59e0b;
        font-weight: 700;
    }
</style>
""", unsafe_allowed_html=True)

st.title("📊 Financial Sentiment Classification Hub")
st.markdown("A deep learning benchmarking system evaluating **RNN, LSTM, and FinBERT** models on financial tweet data.")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/financial-analytics.png", width=90)
st.sidebar.title("Configuration")

# Model Selection
model_options = ["RNN", "LSTM", "FinBERT"]
selected_model_type = st.sidebar.selectbox("Select Model for Inference", model_options)

# Helper to check if model trained
def check_model_trained(model_name):
    model_name = model_name.lower()
    models_dir = config.get("models_dir", "./models")
    if model_name in ["rnn", "lstm"]:
        vocab_exists = os.path.exists(os.path.join(models_dir, f"{model_name}_vocab.json"))
        weights_exists = os.path.exists(os.path.join(models_dir, f"{model_name}_best.pt"))
        return vocab_exists and weights_exists
    elif model_name == "finbert":
        return os.path.exists(os.path.join(models_dir, "finbert_best"))
    return False

model_trained = check_model_trained(selected_model_type)

if model_trained:
    st.sidebar.success(f"✅ {selected_model_type} loaded and ready!")
else:
    st.sidebar.warning(f"⚠️ {selected_model_type} checkpoint not found. Run training/fine-tuning first.")

# Load Predictor if trained
@st.cache_resource
def get_predictor(model_name):
    try:
        return SentimentPredictor(model_name)
    except Exception as e:
        return None

predictor = get_predictor(selected_model_type) if model_trained else None

# Sidebar details
st.sidebar.markdown("---")
st.sidebar.markdown("### Benchmarking Settings")
if selected_model_type in ["RNN", "LSTM"]:
    st.sidebar.code(f"""
Model: {selected_model_type.upper()}
Embedding Dim: {config.get('rnn_lstm', {}).get('embedding_dim', 100)}
Hidden Dim: {config.get('rnn_lstm', {}).get('hidden_dim', 128)}
Layers: {config.get('rnn_lstm', {}).get('num_layers', 2)}
Max Length: {config.get('rnn_lstm', {}).get('max_len', 64)}
    """)
else:
    st.sidebar.code(f"""
Model: FinBERT
Base: ProsusAI/finbert
Batch Size: {config.get('finbert', {}).get('batch_size', 16)}
Learning Rate: {config.get('finbert', {}).get('learning_rate', 2e-5)}
Max Length: {config.get('finbert', {}).get('max_len', 64)}
    """)

# Tabs
tab1, tab2 = st.tabs(["🔮 Live Sentiment Inference", "📈 Experiment Benchmarks"])

with tab1:
    st.header(f"Live Inference with {selected_model_type}")
    
    # Input box
    sample_tweet = st.selectbox(
        "Select a sample tweet / headline to test:",
        [
            "Custom Input...",
            "$AAPL quarterly revenue beats street expectations, stock surges 4% in after-hours trading.",
            "Market volatility hits financial sector as inflation fears trigger a sell-off in growth stocks.",
            "Tesla CEO Elon Musk tweets about plans to optimize Gigafactory production lines.",
            "$SPY closed flat today ahead of tomorrow's federal reserve interest rate decision."
        ]
    )
    
    if sample_tweet == "Custom Input...":
        user_input = st.text_area(
            "Enter financial tweet or headline:",
            placeholder="Type your financial text here...",
            height=100
        )
    else:
        user_input = st.text_area(
            "Enter financial tweet or headline:",
            value=sample_tweet,
            height=100
        )
        
    if st.button("Classify Sentiment", type="primary"):
        if not model_trained:
            st.error(f"Cannot run inference. The model `{selected_model_type}` is not trained yet. Run the training script in your environment first.")
        elif predictor is None:
            st.error(f"Failed to initialize the predictor for `{selected_model_type}`.")
        elif len(user_input.strip()) == 0:
            st.warning("Please enter a valid non-empty text string.")
        else:
            with st.spinner(f"Running {selected_model_type} forward pass..."):
                res = predictor.predict(user_input)
                
            pred_label = res["predicted_label"]
            confidence = res["confidence"]
            probs = res["probabilities"]
            inf_time = res["inference_time_ms"]
            
            # Display Results Cards
            c1, c2, c3 = st.columns(3)
            
            # Select class value style
            val_style = "metric-val-neutral"
            if pred_label == "Bullish":
                val_style = "metric-val-bullish"
            elif pred_label == "Bearish":
                val_style = "metric-val-bearish"
                
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted Sentiment</div>
                    <div class="{val_style}">{pred_label}</div>
                </div>
                """, unsafe_allowed_html=True)
                
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Confidence Score</div>
                    <div class="metric-val-neutral">{confidence * 100:.1f}%</div>
                </div>
                """, unsafe_allowed_html=True)
                
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Inference Speed</div>
                    <div class="metric-val-neutral">{inf_time:.2f} ms</div>
                </div>
                """, unsafe_allowed_html=True)
                
            # Class probabilities visualization
            st.markdown("### Sentiment Probabilities")
            prob_df = pd.DataFrame(
                list(probs.items()), 
                columns=["Sentiment Class", "Probability"]
            ).sort_values("Probability", ascending=False)
            
            st.bar_chart(data=prob_df, x="Sentiment Class", y="Probability", height=250)

with tab2:
    st.header("Benchmark Performance Log")
    
    # Load experiment log
    reports_dir = config.get("reports_dir", "./reports")
    log_path = os.path.join(reports_dir, "experiment_log.csv")
    
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        
        # Display table
        st.dataframe(
            log_df.style.highlight_max(subset=["val_accuracy", "val_macro_f1"], color="#1e3a8a"),
            use_container_width=True
        )
        
        # Plot comparative bar charts
        st.subheader("Model Performance Comparison")
        col_acc, col_f1 = st.columns(2)
        
        with col_acc:
            st.markdown("**Validation Accuracy**")
            st.bar_chart(log_df, x="model", y="val_accuracy", height=300)
            
        with col_f1:
            st.markdown("**Validation Macro F1**")
            st.bar_chart(log_df, x="model", y="val_macro_f1", height=300)
            
    else:
        st.info("No benchmark history found. Once models are trained, metrics logged in `reports/experiment_log.csv` will be displayed here.")
        
    # Displays confusion matrices
    st.subheader("Confusion Matrices")
    fig_dir = config.get("figures_dir", "./figures")
    
    cm_cols = st.columns(3)
    cm_models = ["RNN", "LSTM", "FINBERT"]
    
    for i, m_name in enumerate(cm_models):
        with cm_cols[i]:
            st.markdown(f"**{m_name}**")
            cm_path = os.path.join(fig_dir, f"confusion_matrix_{m_name.lower().replace(' ', '_')}.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"{m_name} Confusion Matrix")
            else:
                st.caption("Confusion matrix image not generated yet. Train this model to render.")
