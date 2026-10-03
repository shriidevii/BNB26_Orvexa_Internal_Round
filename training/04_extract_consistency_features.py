import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_CSV = BASE_DIR / "data" / "cases.csv"
OUTPUT_DIR = BASE_DIR / "features" / "consistency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_consistency_features():
    if not CASES_CSV.exists():
        print(f"[!] {CASES_CSV} not found.")
        return

    df = pd.read_csv(CASES_CSV)
    print(f"[*] Extracting consistency features across {len(df)} cases (memory-safe)...")

    # Fit TF-IDF across all case captions and transcripts
    corpus = df["caption"].dropna().tolist() + df["transcript"].dropna().tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb")
    vectorizer.fit(corpus)

    consistency_records = {}

    for _, row in df.iterrows():
        case_id = row["case_id"]
        img_path = row["image_path"]
        caption = str(row.get("caption", ""))
        transcript = str(row.get("transcript", ""))

        # 1. Caption <-> Transcript Cross-Modal Text Cosine Similarity
        tf_cap = vectorizer.transform([caption])
        tf_tr = vectorizer.transform([transcript])
        cap_tr_sim = float(cosine_similarity(tf_cap, tf_tr)[0][0])

        # 2. Image Complexity vs. Text Alignment
        img_cap_sim = 0.50
        img_tr_sim = 0.50
        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                stat = ImageStat.Stat(im.convert("RGB"))
                var_norm = float(np.mean(stat.var) / 2500.0)
                cap_len_norm = min(1.0, len(caption.split()) / 15.0)
                tr_len_norm = min(1.0, len(transcript.split()) / 15.0)

                img_cap_sim = float(np.clip(0.35 + 0.60 * (1.0 - abs(var_norm - cap_len_norm)), 0.05, 0.98))
                img_tr_sim = float(np.clip(0.35 + 0.60 * (1.0 - abs(var_norm - tr_len_norm)), 0.05, 0.98))

        vec = torch.tensor([img_cap_sim, img_tr_sim, cap_tr_sim], dtype=torch.float32)
        consistency_records[case_id] = {
            "vector": vec,
            "img_cap_sim": img_cap_sim,
            "img_tr_sim": img_tr_sim,
            "cap_tr_sim": cap_tr_sim
        }

    out_path = OUTPUT_DIR / "consistency_features.pt"
    torch.save(consistency_records, out_path)
    print(f"[✓] Saved cross-modal consistency features to {out_path}")

if __name__ == "__main__":
    compute_consistency_features()