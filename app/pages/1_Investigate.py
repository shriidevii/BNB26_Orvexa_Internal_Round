import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Insert repository root and app directory into sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / "app"
for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.pipeline import TrustLayerPipeline
from core.spectral_inspector import compute_fft_spectrum, compute_spectral_energy
from core.custody_exporter import generate_custody_manifest

st.set_page_config(page_title="TrustLayer | Investigation", page_icon="🛡", layout="wide")

# Theme CSS aligned with the espresso/teal palette
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Metric Cards in Espresso Theme */
    [data-testid="stMetric"] {
        background-color: #3A302B !important;
        border: 1px solid rgba(207, 196, 185, 0.25) !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #CBBFB3 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] {
        color: #EDE8E3 !important;
        font-weight: 700 !important;
    }

    /* Primary Button in Teal Accent */
    .stButton > button {
        background-color: #00897B !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.2rem !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
    }

    /* Tab bar pill styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #151514;
        padding: 6px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #CBBFB3;
        font-weight: 500;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3A302B !important;
        color: #EDE8E3 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ TrustLayer: Multimodal Authenticity")
st.markdown("<p style='color: #CBBFB3; font-size: 1rem; margin-top: -12px;'>Evidence, Not Just a Verdict. Cross-modal forensic signal correlation.</p>", unsafe_allow_html=True)
st.divider()

pipeline = TrustLayerPipeline()
cases_file = ROOT_DIR / "data" / "cases.csv"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Forensic Settings")
    use_demo = st.checkbox("Enable Cross-Modal Benchmark", value=True)
    
    img_path, aud_path, caption, transcript, selected_id = None, None, "", "", "MANUAL"
    
    if use_demo and cases_file.exists():
        df = pd.read_csv(cases_file)
        selected_id = st.selectbox("Detection Target:", df["case_id"].tolist())
        case_row = df[df["case_id"] == selected_id].iloc[0]
        img_path = case_row["image_path"]
        aud_path = case_row["audio_path"]
        caption = str(case_row.get("caption", ""))
        transcript = str(case_row.get("transcript", ""))
        
        st.markdown("---")
        st.markdown("<span style='font-size: 0.75rem; color: #CBBFB3; text-transform: uppercase;'>Ground Truth Provenance</span>", unsafe_allow_html=True)
        gt = case_row['ground_truth_case']
        if gt == "AUTHENTIC":
            st.success(gt)
        else:
            st.error(gt)

# --- MEDIA WORKSPACE ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Visual Evidence")
    if img_path and Path(img_path).exists():
        st.image(str(img_path), use_container_width=True)
        st.caption(f"Context: {caption}")
    else:
        st.info("No visual artifact loaded.")

with col_right:
    st.markdown("#### Audio Evidence")
    if aud_path and Path(aud_path).exists():
        st.write("<br>", unsafe_allow_html=True)
        st.audio(str(aud_path))
        st.caption(f"Speech Transcript: \"{transcript}\"")
    else:
        st.info("No acoustic artifact loaded.")

st.write("")

# --- RUN ANALYSIS ---
if st.button("RUN FORENSIC ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Correlating cross-modal forensic signatures..."):
        results = pipeline.analyze(img_path, aud_path, caption, transcript)

    st.write("<br>", unsafe_allow_html=True)

    # 1. Verdict Banner (Transparent Heuristic Override)
    p_img = results['image_expert']['p_synthetic']
    p_aud = results['audio_expert']['p_synthetic']
    q_hat = results['fusion']['q_hat_conformal']
    
    # Calculate transparent average
    p_fusion = (p_img + p_aud) / 2.0
    conflict = abs(p_img - p_aud) > 0.4

    # Apply transparent math: Base 0.5 threshold +/- half the conformal margin
    if conflict:
        verdict = "MANIPULATION SUSPECTED"
        v_desc = "Cross-modal semantic dissonance detected."
    elif p_fusion > (0.5 + (q_hat / 2)):
        verdict = "MANIPULATION SUSPECTED"
        v_desc = "Combined unimodal risks exceed conformal safety thresholds."
    elif p_fusion < (0.5 - (q_hat / 2)):
        verdict = "AUTHENTIC"
        v_desc = "Cross-modal alignment indicates genuine human capture."
    else:
        verdict = "INCONCLUSIVE"
        v_desc = "Sample falls within conformal uncertainty bounds (Review Required)."

    with st.container(border=True):
        if verdict == "AUTHENTIC":
            st.success(f"**VERDICT: {verdict}** — {v_desc}")
        elif verdict == "MANIPULATION SUSPECTED":
            st.error(f"**VERDICT: {verdict}** — {v_desc}")
        else:
            st.warning(f"**VERDICT: {verdict}** — {v_desc}")

        # 2. Metric Tiles (Now 5 columns with Overall Fusion Risk)
        st.write("")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Visual Deepfake Risk", f"{p_img * 100:.1f}%")
        m2.metric("Synthetic Speech Risk", f"{p_aud * 100:.1f}%")
        m3.metric("Overall Fusion Risk", f"{p_fusion * 100:.1f}%")
        m4.metric("Modality Consistency", "Conflict" if conflict else "Aligned")
        m5.metric("Conformal Bound (α=0.1)", f"{q_hat:.3f}")

    st.write("<br>", unsafe_allow_html=True)

    # 3. Telemetry Tabs
    with st.container(border=True):
        t_spectral, t_graph, t_audit = st.tabs(["Forensic Signals", "Evidence Graph", "Raw Evidence"])

        with t_spectral:
            col_spec_img, col_spec_aud = st.columns(2)
            with col_spec_img:
                fig_fft = compute_fft_spectrum(img_path)
                st.pyplot(fig_fft)
            with col_spec_aud:
                fig_aud = compute_spectral_energy(aud_path)
                st.pyplot(fig_aud)

        with t_graph:
            fig, ax = plt.subplots(figsize=(10, 4.2))
            fig.patch.set_facecolor('#231B17')
            ax.set_facecolor('#231B17')

            G = results["graph"]
            pos = nx.spring_layout(G, seed=42, k=0.95)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#3A302B", node_size=2800, edgecolors="#00897B", linewidths=1.8)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#6F6055", width=2, arrows=True, arrowsize=14, connectionstyle="arc3,rad=0.08")
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#EDE8E3", font_weight="bold")
            nx.draw_networkx_edge_labels(
                G, pos, 
                edge_labels=nx.get_edge_attributes(G, 'relation'), 
                font_size=8, font_color="#CBBFB3", 
                bbox=dict(facecolor='#231B17', edgecolor='none')
            )
            ax.axis("off")
            st.pyplot(fig)

        with t_audit:
            manifest_json = generate_custody_manifest(img_path, aud_path, results, case_id=selected_id)
            col_dl, col_man = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    label="📥 Download Forensic Report (.json)",
                    data=manifest_json,
                    file_name=f"TrustLayer_Report_{selected_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_man:
                st.json(json.loads(manifest_json), expanded=True)