import os
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "features" / "audio"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(manifest_csv: str):
    df = pd.read_csv(manifest_csv)
    if df.empty:
        print(f"[!] Manifest {manifest_csv} is empty.")
        return

    # Load pretrained Wav2Vec2 bundle from torchaudio
    print(f"[*] Loading Wav2Vec2 Feature Extractor on {DEVICE}...")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(DEVICE)
    model.eval()

    extracted = {}
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio features"):
            path = row["file_path"]
            if not os.path.exists(path):
                continue
            waveform, sample_rate = torchaudio.load(path)
            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
            
            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.to(DEVICE)
            # Extract representations and mean-pool across time frames -> 768-dim vector
            features, _ = model.extract_features(waveform)
            embedding = torch.mean(features[-1], dim=1).squeeze(0).cpu()
            extracted[row["sample_id"]] = embedding

    output_path = FEATURES_DIR / "audio_embeddings.pt"
    torch.save(extracted, output_path)
    print(f"[+] Saved {len(extracted)} audio representations to {output_path}")

if __name__ == "__main__":
    extract_features(str(BASE_DIR / "data" / "expert_audio.csv"))