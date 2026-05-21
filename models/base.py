import torch
import torch.nn as nn


# ── Base ──────────────────────────────────────────────────────────────────────
 
class BaseClassifier(nn.Module):
    """All models inherit from this. Enforces a common interface."""
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
 
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))