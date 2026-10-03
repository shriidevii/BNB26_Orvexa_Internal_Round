import torch
import torch.nn as nn

class MultimodalFusionMLP(nn.Module):
    def __init__(self, in_features=2051, hidden_dim=256, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)