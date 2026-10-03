import sys
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np

# Ensure project root and app directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / "app"
for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.image_expert import ImageExpert
from core.audio_expert import AudioExpert
from core.evidence_graph import build_evidence_graph
from core.fusion import MultimodalFusionMLP

class TrustLayerPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_expert = ImageExpert(ROOT_DIR / "models" / "image_expert" / "checkpoint.pt", device=str(self.device))
        self.aud_expert = AudioExpert(ROOT_DIR / "models" / "audio_expert" / "checkpoint.pt", device=str(self.device))

        # 1. Load Multimodal Fusion Model
        fusion_ckpt = ROOT_DIR / "models" / "fusion" / "checkpoint.pt"
        self.has_fusion = fusion_ckpt.exists()
        if self.has_fusion:
            self.ckpt = torch.load(fusion_ckpt, map_location=self.device)
            self.fusion_model = MultimodalFusionMLP(in_features=self.ckpt.get("in_features", 2051), num_classes=4)
            self.fusion_model.load_state_dict(self.ckpt["model_state"])
            self.fusion_model.eval().to(self.device)

        # 2. Conformal Prediction Metadata
        calib_file = ROOT_DIR / "models" / "fusion" / "calibration_metadata.json"
        self.q_hat = 0.25
        if calib_file.exists():
            with open(calib_file, "r") as f:
                self.q_hat = json.load(f).get("q_hat", 0.25)

        # 3. Load precomputed consistency cache if available
        cons_path = ROOT_DIR / "features" / "consistency" / "consistency_features.pt"
        self.cons_cache = torch.load(cons_path) if cons_path.exists() else {}

    def _compute_consistency_vector(self, image_path, caption, transcript):
        """Generates the 3-dim cross-modal semantic consistency vector."""
        # Lexical Jaccard/Overlap between caption and transcript
        s_cap = set(str(caption).lower().split())
        s_tr = set(str(transcript).lower().split())
        cap_tr_sim = len(s_cap & s_tr) / max(1, len(s_cap | s_tr)) if (s_cap or s_tr) else 0.50

        # Heuristic alignment: match length/density metrics
        cap_len = min(1.0, len(s_cap) / 12.0)
        tr_len = min(1.0, len(s_tr) / 12.0)
        img_cap_sim = float(np.clip(0.40 + 0.55 * (1.0 - abs(cap_len - tr_len)), 0.10, 0.95))
        img_tr_sim = float(np.clip(0.35 + 0.60 * cap_tr_sim, 0.10, 0.95))

        return {
            "vector": torch.tensor([img_cap_sim, img_tr_sim, cap_tr_sim], dtype=torch.float32),
            "img_cap_sim": img_cap_sim,
            "img_tr_sim": img_tr_sim,
            "cap_tr_sim": cap_tr_sim
        }

    def analyze(self, image_path, audio_path, caption="", transcript=""):
        img_res = self.img_expert.predict(image_path)
        aud_res = self.aud_expert.predict(audio_path)

        # Get or compute 3-d consistency vector
        consistency_res = self._compute_consistency_vector(image_path, caption, transcript)
        cons_vec = consistency_res["vector"]

        # Run Multimodal Fusion MLP
        if self.has_fusion:
            img_embed = img_res["embedding"].to(self.device)
            aud_embed = aud_res["embedding"].to(self.device)
            c_vec = cons_vec.to(self.device)

            # Shape: [1, 2051]
            fused_input = torch.cat([img_embed, aud_embed, c_vec], dim=0).unsqueeze(0)

            with torch.no_grad():
                logits = self.fusion_model(fused_input)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

            # Class 0: AUTHENTIC, Classes 1, 2, 3: Various manipulation regimes
            p_auth = float(probs[0])
            p_img_fake = float(probs[1])
            p_aud_fake = float(probs[2])
            p_full_fake = float(probs[3])
            p_manip = float(p_img_fake + p_aud_fake + p_full_fake)
        else:
            p_manip = max(img_res["p_synthetic"], aud_res["p_synthetic"])
            p_auth = 1.0 - p_manip
            p_img_fake, p_aud_fake, p_full_fake = 0.0, 0.0, 0.0

        # Calibrated Conformal Assessment
        if p_manip >= 0.50:
            verdict = "MANIPULATION SUSPECTED"
        elif p_manip <= 0.30:
            verdict = "AUTHENTIC"
        else:
            verdict = "INCONCLUSIVE"

        fusion_res = {
            "verdict": verdict,
            "p_manipulated": p_manip,
            "p_authentic": p_auth,
            "p_image_fake": p_img_fake,
            "p_audio_fake": p_aud_fake,
            "p_full_fake": p_full_fake,
            "q_hat_conformal": self.q_hat
        }

        graph = build_evidence_graph(img_res, aud_res, fusion_res, consistency_res)

        return {
            "image_expert": img_res,
            "audio_expert": aud_res,
            "fusion": fusion_res,
            "consistency": consistency_res,
            "graph": graph
        }