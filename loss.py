"""
CPC Loss Function
Implements the InfoNCE contrastive predictive coding loss
"""
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
        normalize (bool): Whether to L2-normalize features (default: True)
    """
    def __init__(self, tau=0.07, normalize=True):
        super().__init__()
        self.tau = tau
        self.normalize = normalize

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
        
        # Normalize if requested
        if self.normalize:
            preds = F.normalize(preds, dim=-1)
            targets = F.normalize(targets, dim=-1)
        
        total_loss = 0.0
        total_correct = 0
        
        for i in range(timestep):
            pred_i = preds[i]  # [B, D]
            target_i = targets[i]  # [B, D]
            
            # Compute similarity matrix: [B, D] @ [D, B] -> [B, B]
            # Each row corresponds to one sample's prediction vs all targets
            logits = torch.mm(target_i, pred_i.t()) / self.tau
            
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


class InfoNCELoss(nn.Module):
    """
    Alternative simpler InfoNCE implementation
    Computes loss for a single prediction step
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query, key):
        """
        Args:
            query: Query representations [B, D]
            key: Key representations [B, D]
        
        Returns:
            loss: InfoNCE loss
        """
        # Normalize
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        
        # Compute similarity
        logits = torch.mm(query, key.t()) / self.temperature
        
        # Labels: diagonal is positive
        labels = torch.arange(query.size(0), device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
