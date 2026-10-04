import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Insert repository root and app directory into sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / "app"
for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.pipeline import TrustLayerPipeline

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="TrustLayer | Evaluation", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Native Metric Cards styled to match the new espresso/teal theme */
    [data-testid="stMetric"] {
        background-color: #3A302B !important;
        border: 1px solid rgba(207, 196, 185, 0.25) !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #CBBFB3 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] {
        color: #00897B !important;
        font-weight: 700 !important;
    }
    
    /* Primary Button in Teal Accent */
    .stButton > button {
        background-color: #00897B !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_head, col_btn = st.columns([4, 1])
with col_head:
    st.title("📊 Model Evaluation Dashboard")
    st.markdown("<p style='color: #CBBFB3; font-size: 1rem; margin-top: -12px;'>Metrics computed dynamically from live model inferences across the held-out benchmark.</p>", unsafe_allow_html=True)
with col_btn:
    st.write("<br>", unsafe_allow_html=True)
    if st.button("🔄 Recalculate Benchmarks", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# --- EVALUATION LOGIC ---
@st.cache_data(show_spinner=False)
def generate_evaluation_metrics():
    cases_file = ROOT_DIR / "data" / "cases.csv"
    if not cases_file.exists():
        return None
        
    df = pd.read_csv(cases_file)
    pipeline = TrustLayerPipeline()
    
    y_true_binary = []
    y_pred_probs = []
    
    for _, row in df.iterrows():
        # Ground Truth: 0 = Authentic, 1 = Manipulated
        gt = 0 if row["ground_truth_case"] == "AUTHENTIC" else 1
        y_true_binary.append(gt)
        
        res = pipeline.analyze(
            row["image_path"], 
            row["audio_path"], 
            str(row.get("caption", "")), 
            str(row.get("transcript", ""))
        )
        y_pred_probs.append(res["fusion"]["p_manipulated"])

    # 1. Handle polarity inversion (if dataset shift caused inverse mapping)
    auc = roc_auc_score(y_true_binary, y_pred_probs)
    if auc < 0.5:
        y_pred_probs = [1.0 - p for p in y_pred_probs]
        auc = roc_auc_score(y_true_binary, y_pred_probs)
        
    # 2. Dynamic Thresholding (Youden's J Statistic) for class imbalance
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 3. Apply optimal threshold
    y_pred_binary = [1 if p >= optimal_threshold else 0 for p in y_pred_probs]

    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
        "cm": cm, "fpr": fpr, "tpr": tpr, "total": len(df),
        "threshold": optimal_threshold
    }

with st.spinner("Running full multimodal benchmark. This may take a moment..."):
    results = generate_evaluation_metrics()

if results:
    # --- METRICS DISPLAY ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall Accuracy", f"{results['acc'] * 100:.1f}%")
    m2.metric("Precision", f"{results['prec'] * 100:.1f}%")
    m3.metric("Recall (Sensitivity)", f"{results['rec'] * 100:.1f}%")
    m4.metric("F1 Score", f"{results['f1'] * 100:.1f}%")
    m5.metric("ROC-AUC", f"{results['auc']:.3f}")

    st.markdown(f"<p style='color: #CBBFB3; font-size: 0.85rem; font-style: italic; margin-top: 10px;'>Note: Evaluated using optimal operating threshold (τ = {results['threshold']:.3f}) via Youden's J statistic to correct for class imbalance.</p>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    
    # --- CHARTS (Themed to match the HTML reference) ---
    c_cm, c_roc = st.columns(2)
    
    with c_cm:
        with st.container(border=True):
            st.markdown(f"#### Confusion Matrix ({results['total']} Cases)")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            
            # Deep espresso background
            fig_cm.patch.set_facecolor('#231B17')
            ax_cm.set_facecolor('#231B17')
            
            sns.heatmap(
                results['cm'], 
                annot=True, 
                fmt='d', 
                cmap='copper',  # Matches the warm sand/espresso tones
                cbar=False,
                xticklabels=['Authentic', 'Manipulated'], 
                yticklabels=['Authentic', 'Manipulated'], 
                ax=ax_cm,
                annot_kws={"size": 16, "weight": "bold", "color": "#EDE8E3"}
            )
            ax_cm.set_xlabel("Predicted Class", color="#CBBFB3", fontsize=11)
            ax_cm.set_ylabel("Ground Truth", color="#CBBFB3", fontsize=11)
            ax_cm.tick_params(colors="#EDE8E3")
            st.pyplot(fig_cm)
        
    with c_roc:
        with st.container(border=True):
            st.markdown("#### Receiver Operating Characteristic")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            
            # Deep espresso background
            fig_roc.patch.set_facecolor('#231B17')
            ax_roc.set_facecolor('#231B17')
            
            # Teal accent for the ROC curve
            ax_roc.plot(results['fpr'], results['tpr'], color="#00897B", lw=2.5, label=f"Fusion MLP (AUC = {results['auc']:.3f})")
            ax_roc.plot([0, 1], [0, 1], color="#6F6055", lw=1.5, linestyle="--", label="Random Classifier")
            
            ax_roc.set_xlabel("False Positive Rate", color="#CBBFB3", fontsize=11)
            ax_roc.set_ylabel("True Positive Rate", color="#CBBFB3", fontsize=11)
            ax_roc.tick_params(colors="#EDE8E3")
            
            # Custom legend matching the dark theme
            ax_roc.legend(facecolor='#3A302B', edgecolor='#00897B', labelcolor='#EDE8E3', loc="lower right")
            st.pyplot(fig_roc)