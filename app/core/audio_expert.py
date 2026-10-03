import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np

class AudioExpert:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.backbone = self.bundle.get_model().to(self.device)
        self.backbone.eval()

        self.head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.head.load_state_dict(ckpt["model_state"])
        self.head.eval().to(self.device)

    def predict(self, aud_path):
        # 100% bypass of torchaudio loader to fix the torchcodec crash
        wav_np, sr = sf.read(aud_path, dtype='float32')
        
        # Format numpy array for PyTorch
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]
        else:
            wav_np = wav_np.T
            
        wav = torch.from_numpy(wav_np)
        
        # Resample and format channels
        if sr != self.bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.bundle.sample_rate)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        wav = wav.to(self.device)

        with torch.no_grad():
            feats, _ = self.backbone.extract_features(wav)
            feat = torch.mean(feats[-1], dim=1)
            logits = self.head(feat)
            probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
            
        return {
            "p_authentic": float(probs[0]),
            "p_synthetic": float(probs[1]),
            "embedding": feat.squeeze(0).cpu()
        }