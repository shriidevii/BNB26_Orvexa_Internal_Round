import sys
from pathlib import Path

# Insert repository root and app directory into sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / "app"
for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from core.pipeline import TrustLayerPipeline

st.set_page_config(page_title="Investigate | TrustLayer", page_icon="🔍", layout="wide")
st.title("🔍 Multimodal Artifact Investigation")

pipeline = TrustLayerPipeline()
cases_file = ROOT_DIR / "data" / "cases.csv"

st.sidebar.header("Select or Upload Case")
use_demo = st.sidebar.checkbox("Load Controlled Benchmark Case", value=True)

img_path, aud_path, caption, transcript = None, None, "", ""

if use_demo and cases_file.exists():
    df = pd.read_csv(cases_file)
    selected_id = st.sidebar.selectbox("Benchmark Cases", df["case_id"].tolist())
    case_row = df[df["case_id"] == selected_id].iloc[0]
    img_path = case_row["image_path"]
    aud_path = case_row["audio_path"]
    caption = str(case_row.get("caption", ""))
    transcript = str(case_row.get("transcript", ""))
    st.sidebar.caption(f"Ground Truth Provenance: **{case_row['ground_truth_case']}**")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Visual Artifact")
    if img_path and Path(img_path).exists():
        st.image(str(img_path), use_container_width=True)
    st.caption(f"Caption: {caption}")

with col_right:
    st.subheader("Audio Artifact")
    if aud_path and Path(aud_path).exists():
        st.audio(str(aud_path))
    st.caption(f"Transcript: {transcript}")

if st.button("🚀 START INVESTIGATION", type="primary"):
    with st.spinner("Executing unimodal experts and cross-modal reasoning..."):
        results = pipeline.analyze(img_path, aud_path, caption, transcript)

    verdict = results["fusion"]["verdict"]
    if verdict == "AUTHENTIC":
        st.success(f"### Assessment: {verdict}")
    elif verdict == "MANIPULATION SUSPECTED":
        st.error(f"### Assessment: {verdict}")
    else:
        st.warning(f"### Assessment: {verdict}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Visual AI Synthetic Prob", f"{results['image_expert']['p_synthetic'] * 100:.1f}%")
    m2.metric("Audio AI Synthetic Prob", f"{results['audio_expert']['p_synthetic'] * 100:.1f}%")
    m3.metric(
        "Multimodal Conflict",
        "Contradiction Detected"
        if abs(results["image_expert"]["p_synthetic"] - results["audio_expert"]["p_synthetic"]) > 0.4
        else "Modality Agreement",
    )
    m4.metric("Conformal Uncertainty Margin", f"{results['fusion']['q_hat_conformal']:.3f}")

    st.markdown("---")
    st.markdown("---")
    st.subheader("🕸️ Dynamic Forensic Evidence Graph")
    
    # Upgraded Professional Dark-Themed Graph
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0E1117') # Match Streamlit dark theme
    ax.set_facecolor('#0E1117')

    G = results["graph"]
    # Increase 'k' to spread nodes out further
    pos = nx.spring_layout(G, seed=42, k=0.9) 

    # Draw Nodes with custom styling
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#1E293B", node_size=2800, 
                           edgecolors="#38BDF8", linewidths=2)
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#475569", width=2, 
                           arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1")
    
    # Draw Node Labels (White text)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white", font_weight="bold")
    
    # Draw Edge Labels (The relationships)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                 font_color="#94A3B8", bbox=dict(facecolor='#0E1117', edgecolor='none'))
    
    ax.axis("off")
    st.pyplot(fig)