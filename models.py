"""EfficientNet-B0 classifier — architecture must match
Experimenting/classifier_experiments/common/model_b0.py so the trained
state_dict loads cleanly."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

B0_FEATURE_DIM = 1280
HEAD_HIDDEN = 256
DROPOUT = 0.1


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        net = efficientnet_b0(weights=None)
        net.classifier = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(B0_FEATURE_DIM, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_classifier(weights_path, device: torch.device) -> EfficientNetB0Classifier:
    model = EfficientNetB0Classifier().to(device)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
