"""
CPC Loss Function
Implements the InfoNCE contrastive predictive coding loss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCLoss(nn.Module):
    """
    Contrastive Predictive Coding Loss (InfoNCE)
    
    For each time step k in the future, we predict z_{t+k} from context c_t
    using linear transformations W_k. The loss maximizes mutual information
    between predictions and future observations.
    
    Args:
        tau (float): Temperature parameter for softmax (default: 0.07)
    """
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau
        print(f"CPCLoss: tau={tau:.4f}")

    def forward(self, preds, targets):
        """
        Compute CPC loss across multiple time steps
        
        Args:
            preds: Predicted future representations [timestep, B, D]
            targets: Ground truth future representations [timestep, B, D]
            
        Returns:
            loss: Average InfoNCE loss across time steps
            accuracy: Prediction accuracy (diagnostic metric)
        """
        timestep, B, D = preds.shape
        
        scale = 1.0 / math.sqrt(D)
        
        total_loss = 0.0
        total_correct = 0
        
        for i in range(timestep):
            pred_i = preds[i]  # [B, D]
            target_i = targets[i]  # [B, D]
            
            # Compute similarity matrix: [B, D] @ [D, B] -> [B, B]
            # Apply dimension scaling before temperature
            logits = torch.mm(target_i, pred_i.t()) * scale / self.tau
            
            # Diagonal elements are positive pairs
            labels = torch.arange(B, device=preds.device)
            
            # InfoNCE: cross-entropy where positives are on diagonal
            loss = F.cross_entropy(logits, labels, reduction="mean")
            total_loss += loss
            
            # Accuracy: how often is the correct target the highest score?
            pred_labels = logits.argmax(dim=1)
            total_correct += (pred_labels == labels).sum().item()
        
        # Average over time steps
        avg_loss = total_loss / timestep
        accuracy = total_correct / (B * timestep)
        
        return avg_loss, accuracy
