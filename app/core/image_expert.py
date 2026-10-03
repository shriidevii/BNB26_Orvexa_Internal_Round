import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageExpert:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.backbone.eval().to(self.device)

        self.head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.head.load_state_dict(ckpt["model_state"])
        self.head.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(tensor)
            logits = self.head(feat)
            probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        return {
            "p_authentic": float(probs[0]),
            "p_synthetic": float(probs[1]),
            "embedding": feat.squeeze(0).cpu()
        }