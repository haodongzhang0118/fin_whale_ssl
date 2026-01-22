"""
CPC Backbone for Fin-Whale Audio
Implements the CDCK2 architecture with SincNet and BRN normalization
Compatible with stable-pretraining framework
"""
import torch
import torch.nn as nn
from sincNet import SincBlock

# Batch-RMS Normalization
class BRN1d(nn.Module):
    """
    Batch-RMS Normalization for 1D [B, C, T]
    y = (rho * BN(x) + (1 - rho) * RMSN(x)) * gamma + beta
    
    Uses PyTorch official nn.RMSNorm (available in PyTorch 2.4+)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, rho_init=0.5, affine=True):
        super().__init__()
        self.num_features = num_features
        
        # BatchNorm1d for the BN component
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum,
                                 affine=False, track_running_stats=True)
        
        # RMSNorm normalizes over the last dimension, so we need to handle [B, C, T] -> [B, T, C]
        self.rms = nn.RMSNorm(num_features, eps=eps, elementwise_affine=False)
        
        # Learnable mixing parameter
        self.rho = nn.Parameter(torch.full((1, num_features, 1), float(rho_init)))

        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, T]
        Returns:
            y: Normalized tensor [B, C, T]
        """
        # BatchNorm component (operates on C dimension)
        x_bn = self.bn(x)  # [B, C, T]
        
        # Official RMSNorm normalizes the last dimension, so transpose: [B, C, T] -> [B, T, C]
        x_transposed = x.transpose(1, 2)  # [B, T, C]
        x_rms_transposed = self.rms(x_transposed)  # [B, T, C]
        x_rms = x_rms_transposed.transpose(1, 2)  # [B, C, T]
        
        # Mix BatchNorm and RMSNorm with learnable rho
        rho = self.rho.clamp(0.0, 1.0)
        y = rho * x_bn + (1 - rho) * x_rms
        
        # Apply affine transformation
        if self.affine:
            y = y * self.gamma + self.beta
        
        return y


class CPCEncoder(nn.Module):
    """
    Encoder network: stacked Conv1d + BRN + ReLU
    Downsamples the SincNet output to produce latent representations
    
    Architecture with 6 conv layers for aggressive downsampling:
    - Total encoder downsampling: 5 × 3 × 2 × 2 × 1 × 1 = 60x
    - Combined with SincBlock (5x): 300x total
    - For 16000 Hz, 8s audio: 128000 samples → 427 time steps
    """
    def __init__(self, in_chan=512):
        super().__init__()
        self.encoder = nn.Sequential(
            # Layer 1: stride=5
            nn.Conv1d(in_chan, 512, kernel_size=8, stride=5, padding=2, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            # Layer 2: stride=3
            nn.Conv1d(512, 512, kernel_size=4, stride=3, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            # Layer 3: stride=2
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            # Layer 4: stride=2
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            # Layer 5: stride=1
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            # Layer 6: stride=1
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class CPCBackbone(nn.Module):
    """
    CPC Backbone Network for audio contrastive learning
    
    Architecture:
        Input -> SincBlock -> Encoder -> GRU -> Context vectors
    
    Args:
        sample_rate (int): Audio sample rate (default: 3200 Hz for fin-whale)
        enc_hidden (int): Encoder output dimension (default: 512)
        gru_hidden (int): GRU hidden dimension (default: 256)
        sinc_channels (int): SincNet output channels (default: 512)
    """
    def __init__(
        self, 
        sample_rate=3200, 
        enc_hidden=512, 
        gru_hidden=256,
        sinc_channels=512
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.enc_hidden = enc_hidden
        self.gru_hidden = gru_hidden
        
        # SincNet front-end for learnable filterbank
        self.sinc_block = SincBlock(
            out_channels=sinc_channels, 
            stride=5, 
            sample_rate=sample_rate,
            return_abs=False, 
            learnable_filters=False, 
            padding="same"
        )
        
        # Encoder: converts SincNet features to latent representations z_t
        self.encoder = CPCEncoder(in_chan=sinc_channels)
        
        # GRU autoregressor: summarizes past context
        self.gru = nn.GRU(
            enc_hidden, 
            gru_hidden, 
            num_layers=1, 
            bidirectional=False, 
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass producing both encoded features and context vectors
        
        Args:
            x: Input waveform [B, 1, T] or [B, T]
            
        Returns:
            c_t: Context vectors from GRU [B, T', gru_hidden]
            z_t: Encoded features [B, T', enc_hidden]
        """
        # Ensure input is [B, 1, T]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        # SincNet filtering
        sinc_out = self.sinc_block(x)  # [B, sinc_channels, T']
        
        # Encoder
        z = self.encoder(sinc_out)  # [B, enc_hidden, T'']
        z = z.transpose(1, 2)  # [B, T'', enc_hidden]
        
        # GRU for context
        c_t, _ = self.gru(z)  # [B, T'', gru_hidden]
        
        return c_t, z

    def extract_embedding(self, x):
        """
        Extract fixed-size embedding by averaging context vectors
        Used during validation/testing for downstream tasks
        
        Args:
            x: Input waveform [B, 1, T] or [B, T]
            
        Returns:
            embedding: Fixed-size embedding [B, gru_hidden]
        """
        with torch.no_grad():
            c_t, _ = self.forward(x)
            # Average pool over time
            embedding = c_t.mean(dim=1)  # [B, gru_hidden]
        return embedding


def cpc_backbone(sample_rate=3200, **kwargs):
    """
    Factory function for CPC backbone
    
    Args:
        sample_rate (int): Audio sample rate
        **kwargs: Additional arguments for CPCBackbone
        
    Returns:
        CPCBackbone instance
    """
    return CPCBackbone(sample_rate=sample_rate, **kwargs)
