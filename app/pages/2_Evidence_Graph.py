import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Evidence Graph | TrustLayer", page_icon="🕸️", layout="wide")
st.title("🕸️ Forensic Evidence Graph Architecture")

st.markdown("""
### Mapping Modality Conflicts
TrustLayer does not just output a probability score; it builds a **Directed Evidence Graph** using `NetworkX`. 
This allows human investigators to visually trace exactly *why* a piece of media was flagged, mapping the relationships and contradictions between visual features, acoustic representations, and semantic text.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.info("### Edge Relationships")
    st.markdown("""
    * **`evaluated_by`**: Links raw media to its respective AI Expert model.
    * **`inputs_evidence`**: Feeds unimodal predictions into the Multimodal Fusion MLP.
    * **`modality_agreement`**: Created when Visual and Audio experts agree (e.g., both Authentic).
    * **`modality_contradiction`**: Triggered when one modality is flagged as Synthetic, but the other passes as Authentic.
    * **`semantic_conflict`**: (CLIP/Wav2Vec) Triggered when the spoken transcript contradicts the visual context.
    """)

with col2:
    st.subheader("Example: Synthetic Audio Overlay (Deepfake Voice)")
    # Generate a dummy graph to demonstrate the architecture
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    G = nx.DiGraph()
    G.add_node("Image", type="artifact")
    G.add_node("Audio", type="artifact")
    G.add_node("Image Expert", type="model")
    G.add_node("Audio Expert", type="model")
    G.add_node("Multimodal Fusion", type="model")
    G.add_node("Verdict (FAKE)", type="verdict")

    G.add_edge("Image", "Image Expert", relation="evaluated_by")
    G.add_edge("Audio", "Audio Expert", relation="evaluated_by")
    G.add_edge("Image Expert", "Multimodal Fusion", relation="Authentic (0.12)")
    G.add_edge("Audio Expert", "Multimodal Fusion", relation="Synthetic (0.94)")
    G.add_edge("Image Expert", "Audio Expert", relation="Modality Contradiction!")
    G.add_edge("Multimodal Fusion", "Verdict (FAKE)", relation="concludes")

    pos = nx.spring_layout(G, seed=42, k=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#1E293B", node_size=2500, edgecolors="#F43F5E", linewidths=2)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#475569", width=2, arrows=True, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white", font_weight="bold")
    
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="#94A3B8", bbox=dict(facecolor='#0E1117', edgecolor='none'))
    
    ax.axis("off")
    st.pyplot(fig)