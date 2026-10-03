import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_MODEL_DIR = BASE_DIR / "models" / "image_expert"
AUD_MODEL_DIR = BASE_DIR / "models" / "audio_expert"
IMG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
AUD_MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ExpertHead(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2) # [logit_authentic, logit_synthetic]
        )
    def forward(self, x):
        return self.net(x)

def train_head(embeddings_path, manifest_path, in_dim, save_dir, name):
    print(f"\n[*] Training {name} Classifier Head...")
    embeds = torch.load(embeddings_path)
    df = pd.read_csv(manifest_path)
    
    X, y = [], []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        if sid in embeds:
            X.append(embeds[sid])
            y.append(int(row["label"]))

    if not X:
        print(f"[!] No matching features found for {name}.")
        return

    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)

    model = ExpertHead(in_features=in_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(25):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1)
        acc = (preds == y).float().mean().item()
        print(f"[✓] {name} Training Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}%")

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": in_dim,
        "classes": ["AUTHENTIC", "SYNTHETIC"]
    }
    torch.save(checkpoint, Path(save_dir) / "checkpoint.pt")
    print(f"[+] Saved model checkpoint to {save_dir}/checkpoint.pt")

if __name__ == "__main__":
    train_head(
        BASE_DIR / "features" / "image" / "image_embeddings.pt",
        BASE_DIR / "data" / "expert_image.csv",
        1280, IMG_MODEL_DIR, "Image Expert (EfficientNet-B0)"
    )
    train_head(
        BASE_DIR / "features" / "audio" / "audio_embeddings.pt",
        BASE_DIR / "data" / "expert_audio.csv",
        768, AUD_MODEL_DIR, "Audio Expert (Wav2Vec2)"
    )