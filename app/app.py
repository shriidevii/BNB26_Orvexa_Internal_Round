import streamlit as st

st.set_page_config(page_title="TrustLayer", page_icon="🛡️", layout="wide")

st.title("🛡️ TrustLayer: Multimodal Digital Authenticity & Trust")
st.markdown("""
### Evidence, Not Just a Verdict.
TrustLayer authenticates digital media through multimodal cross-validation. Rather than analyzing modalities in isolation, TrustLayer correlates visual deepfake indicators, synthetic speech acoustic representations, and cross-modal semantic consistency to deliver calibrated, forensic findings.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Visual Expert", "EfficientNet-B0", "Transfer Head")
col2.metric("Audio Expert", "Wav2Vec2", "Penultimate Pool")
col3.metric("Multimodal Reasoning", "Conformal Prediction", "90% Confidence")

st.info("👈 Select **1_Investigate** in the sidebar to run live multi-expert analysis on paired artifacts.")