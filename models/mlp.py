import torch
import torch.nn as nn

from models.base import BaseClassifier

# ── MLP ───────────────────────────────────────────────────────────────────────
 
class MLP(BaseClassifier):
    """
    Fully-connected MLP for tabular binary classification.
    Configuration is now internally hardcoded.
    """
 
    def __init__(self, input_dim: int):
        super().__init__()
 
        # Hardcoded configuration parameters internal to the class
        hidden_dims = [128, 64, 32]
        dropout = 0.3
 
        layers = []
        in_dim = input_dim
        
        # Automatically building the sequential network block
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
            
        layers.append(nn.Linear(in_dim, 1))  # binary output (logit)
 
        self.net = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
 