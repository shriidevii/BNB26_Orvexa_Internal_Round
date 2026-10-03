import networkx as nx

def build_evidence_graph(image_res, audio_res, fusion_res, consistency_res):
    G = nx.DiGraph()

    # Core Artifact Nodes
    G.add_node("Image", type="artifact")
    G.add_node("Audio", type="artifact")
    G.add_node("Image Expert", type="model", p_fake=round(image_res["p_synthetic"], 3))
    G.add_node("Audio Expert", type="model", p_fake=round(audio_res["p_synthetic"], 3))
    G.add_node("Multimodal Fusion", type="model", verdict=fusion_res["verdict"])
    G.add_node("Verdict", type="verdict", result=fusion_res["verdict"])

    # Edges
    G.add_edge("Image", "Image Expert", relation="evaluated_by")
    G.add_edge("Audio", "Audio Expert", relation="evaluated_by")
    G.add_edge("Image Expert", "Multimodal Fusion", relation="inputs_evidence")
    G.add_edge("Audio Expert", "Multimodal Fusion", relation="inputs_evidence")
    G.add_edge("Multimodal Fusion", "Verdict", relation="concludes")

    # Cross-modal agreement or contradiction logic
    img_fake = image_res["p_synthetic"] > 0.5
    aud_fake = audio_res["p_synthetic"] > 0.5

    if img_fake == aud_fake:
        G.add_edge("Image Expert", "Audio Expert", relation="modality_agreement", weight=1.0)
    else:
        G.add_edge("Image Expert", "Audio Expert", relation="modality_contradiction", weight=-1.0)

    if consistency_res.get("img_tr_sim", 1.0) < 0.15:
        G.add_node("Semantic Conflict", type="anomaly")
        G.add_edge("Image", "Semantic Conflict", relation="inconsistent_with_audio_transcript")

    return G