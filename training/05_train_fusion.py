import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
for p in [str(BASE_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.fusion import MultimodalFusionMLP

CASES_CSV = BASE_DIR / "data" / "cases.csv"
CONSISTENCY_FILE = BASE_DIR / "features" / "consistency" / "consistency_features.pt"
SAVE_DIR = BASE_DIR / "models" / "fusion"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "AUTHENTIC": 0,
    "IMAGE_MANIPULATED": 1,
    "AUDIO_MANIPULATED": 2,
    "MULTIMODAL_SYNTHETIC": 3
}

def train_fusion():
    if not CASES_CSV.exists() or not CONSISTENCY_FILE.exists():
        print("[!] Required cases manifest or consistency features missing.")
        return

    df = pd.read_csv(CASES_CSV)
    consistency_data = torch.load(CONSISTENCY_FILE)

    print(f"[*] Loading feature extractors on {DEVICE}...")
    img_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    img_backbone.classifier = nn.Identity()
    img_backbone.eval().to(DEVICE)

    aud_bundle = torchaudio.pipelines.WAV2VEC2_BASE
    aud_backbone = aud_bundle.get_model().to(DEVICE)
    aud_backbone.eval()

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    X_list, y_list = [], []

    print(f"[*] Building multimodal feature vectors for {len(df)} cases...")
    with torch.no_grad():
        for _, row in df.iterrows():
            cid = row["case_id"]
            img_p, aud_p = row["image_path"], row["audio_path"]
            label_str = row["ground_truth_case"]

            if not os.path.exists(img_p) or not os.path.exists(aud_p) or cid not in consistency_data:
                continue

            # Visual Embedding (1280-dim)
            with Image.open(img_p) as img:
                img_t = img_transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
                img_feat = img_backbone(img_t).squeeze(0).cpu()

            # Audio Embedding via soundfile (768-dim)
            wav_np, sr = sf.read(aud_p, dtype='float32')
            if wav_np.ndim == 1:
                wav_np = wav_np[np.newaxis, :]
            else:
                wav_np = wav_np.T
            wav_t = torch.from_numpy(wav_np)
            if sr != aud_bundle.sample_rate:
                wav_t = torchaudio.functional.resample(wav_t, sr, aud_bundle.sample_rate)
            if wav_t.shape[0] > 1:
                wav_t = torch.mean(wav_t, dim=0, keepdim=True)
            wav_t = wav_t.to(DEVICE)

            feats, _ = aud_backbone.extract_features(wav_t)
            aud_feat = torch.mean(feats[-1], dim=1).squeeze(0).cpu()

            # Consistency Vector (3-dim)
            cons_feat = consistency_data[cid]["vector"]

            fused = torch.cat([img_feat, aud_feat, cons_feat], dim=0)
            X_list.append(fused)
            y_list.append(LABEL_MAP[label_str])

    if not X_list:
        print("[!] No cases could be processed.")
        return

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.long)

    model = MultimodalFusionMLP(in_features=X.shape[1], hidden_dim=256, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1)
        acc = (preds == y).float().mean().item()
        print(f"[✓] Multimodal Fusion Model Trained | Accuracy: {acc * 100:.2f}% | Loss: {loss.item():.4f}")

    checkpoint_path = SAVE_DIR / "checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "in_features": X.shape[1],
        "label_map": LABEL_MAP,
        "classes": list(LABEL_MAP.keys())
    }, checkpoint_path)
    print(f"[+] Saved model checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train_fusion()