import torch
import torch.nn as nn

from models.base import BaseClassifier


# ── CNN ───────────────────────────────────────────────────────────────────────
 
class CNN(BaseClassifier):
    """
    1-D CNN for tabular binary classification.
    Configuration is now internally hardcoded.
    """
 
    def __init__(self, input_dim: int):
        super().__init__()
 
        # Hardcoded configuration parameters internal to the class
        channels = [64, 128, 64]
        kernel_size = 3
        dropout = 0.3
        fc_dims = [64, 32]
 
        # Conv blocks — input shape: (B, 1, input_dim)
        conv_layers = []
        in_ch = 1
        for out_ch in channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
 
        # Global average pooling → (B, channels[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)
 
        # FC head
        fc_layers = []
        in_dim = channels[-1]
        for h in fc_dims:
            fc_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        fc_layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*fc_layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)            # (B, F) → (B, 1, F)
        x = self.conv(x)              # (B, C, F)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.fc(x).squeeze(-1)