"""
CPC Forward Function
Training logic for Contrastive Predictive Coding
Compatible with stable-pretraining framework
"""
import torch


def cpc_forward(self, batch, stage):
    """
    Forward pass for CPC training
    
    This function implements the CPC training procedure:
    1. Extract encoded features z_t and context c_t from backbone
    2. Sample a random time step t
    3. Predict future representations z_{t+k} for k=1,...,K
    4. Compute contrastive loss between predictions and ground truth
    
    Args:
        self: Module instance with the following attributes:
            - backbone: CPCBackbone network
            - Wk: ModuleList of linear predictors (one per future step)
            - cpc_loss: CPCLoss module
            - timestep: Number of future steps to predict
        batch: Dictionary containing:
            - 'raw_audio': Waveform tensor [B, T] or [B, 1, T]
            - 'label': Optional labels for evaluation
        stage: Training stage ('fit', 'validate', 'test')
    
    Returns:
        Dictionary containing:
            - 'loss': CPC loss (training only)
            - 'CPC_acc': Prediction accuracy (training only)
            - 'embedding': Fixed-size representation for downstream tasks
            - 'label': Labels if present in batch
    """
    out = {}
    
    # Get raw audio from batch
    if "raw_audio" in batch:
        x = batch["raw_audio"]
    elif "audio" in batch:
        x = batch["audio"]
    elif "waveform" in batch:
        x = batch["waveform"]
    else:
        raise KeyError(f"Expected 'raw_audio', 'audio', or 'waveform' in batch, got keys: {batch.keys()}")
    
    # Ensure input is [B, 1, T]
    if x.ndim == 2:
        x = x.unsqueeze(1)
    
    batch_size = x.size(0)
    
    # Training: compute CPC loss
    if stage == "fit":
        # Forward through backbone
        c_t, z_t = self.backbone(x)  # c_t: [B, T', D_c], z_t: [B, T', D_z]
        
        seq_len = z_t.size(1)
        timestep = self.timestep
        
        # Randomly sample a time step t (leave room for future predictions)
        t_samples = torch.randint(
            low=0, 
            high=seq_len - timestep, 
            size=(1,), 
            device=x.device
        ).long().item()
        
        # Extract future target representations z_{t+k}
        encode_samples = torch.empty(
            (timestep, batch_size, z_t.size(-1)), 
            device=x.device,
            dtype=z_t.dtype
        )
        for k in range(1, timestep + 1):
            encode_samples[k - 1] = z_t[:, t_samples + k, :]  # [B, D_z]
        
        # Extract context vector at time t
        # Due to GRU/causal attention, c_t[:, t, :] only contains past information
        context = c_t[:, t_samples, :]  # [B, D_c]
        
        # Predict future representations using learned linear maps W_k
        predictions = torch.empty(
            (timestep, batch_size, z_t.size(-1)), 
            device=x.device,
            dtype=z_t.dtype
        )
        for k in range(timestep):
            predictions[k] = self.Wk[k](context)  # [B, D_z]
        
        # Compute CPC loss
        loss, accuracy = self.cpc_loss(predictions, encode_samples)
        
        out["loss"] = loss
        out["CPC_acc"] = accuracy
        
        # Log accuracy to progress bar
        self.log(
            f"{stage}/CPC_acc", 
            accuracy, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True
        )
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
    
    # Validation/Testing: just extract fixed-size embeddings
    else:
        with torch.no_grad():
            # Use the backbone's extract_embedding method
            # This returns a fixed-size representation
            embedding = self.backbone.extract_embedding(x)  # [B, D_c]
            out["embedding"] = embedding
    
    # Include labels if present (for probes)
    if "label" in batch:
        label = batch["label"]
        if torch.is_tensor(label):
            label = label.to(x.device)
        out["label"] = label
    
    return out
