"""
CPC Backbone for Fin-Whale Audio
Implements the CDCK2 architecture with SincNet and BRN normalization
Compatible with stable-pretraining framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sincNet import SincBlock
from stable_pretraining.backbone.vit import TransformerBlock, Attention
from stable_pretraining.backbone.pos_embed import get_1d_sincos_pos_embed

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
    - Total encoder downsampling: 4 × 4 × 2 × 2 × 1 × 1 = 64x
    - Combined with SincBlock (5x): 320x total
    - For 16000 Hz, 8s audio: 128000 samples → 400 time steps
    
    Args:
        in_chan (int): Input channels from SincNet (default: 512)
        enc_hidden (int): Output hidden dimension (default: 512)
    """
    def __init__(self, in_chan=512, enc_hidden=512):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     # Layer 1: stride=4 (4x compression)
        #     nn.Conv1d(in_chan, enc_hidden, kernel_size=8, stride=4, padding=2, bias=False),
        #     BRN1d(enc_hidden),
        #     nn.ReLU(inplace=True),
        #     # Layer 2: stride=4 (16x compression)
        #     nn.Conv1d(enc_hidden, enc_hidden, kernel_size=8, stride=4, padding=2, bias=False),
        #     BRN1d(enc_hidden),
        #     nn.ReLU(inplace=True),
        #     # Layer 3: stride=2 (32x compression)
        #     nn.Conv1d(enc_hidden, enc_hidden, kernel_size=4, stride=2, padding=1, bias=False),
        #     BRN1d(enc_hidden),
        #     nn.ReLU(inplace=True),
        #     # Layer 4: stride=2 (64x compression)
        #     nn.Conv1d(enc_hidden, enc_hidden, kernel_size=4, stride=2, padding=1, bias=False),
        #     BRN1d(enc_hidden),
        #     nn.ReLU(inplace=True),
        #     # Layer 5: stride=1 (maintain 64x)
        #     nn.Conv1d(enc_hidden, enc_hidden, kernel_size=3, stride=1, padding=1, bias=False),
        #     BRN1d(enc_hidden),
        #     nn.ReLU(inplace=True),
        # )

        self.encoder = nn.Sequential(
            nn.Conv1d(in_chan, 512, kernel_size=8, stride=5, padding=2, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=3, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BRN1d(512),
            nn.ReLU(inplace=True)
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
        self.encoder = CPCEncoder(in_chan=sinc_channels, enc_hidden=enc_hidden)
        
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
            embedding: Fixed-size embedding [B, gru_hidden], L2-normalized
        """
        with torch.no_grad():
            c_t, _ = self.forward(x)
            # Average pool over time
            embedding = c_t.mean(dim=1)  # [B, gru_hidden]
            # L2 normalize for better downstream performance
            embedding = F.normalize(embedding, p=2, dim=1)
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

class CausalAttention(Attention):
    """
    Causal wrapper for stable-pretraining's Attention.
    
    Simply adds is_causal=True to scaled_dot_product_attention call.
    Inherits all other functionality from the original Attention class.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor [B, N, D]
            
        Returns:
            Output tensor [B, N, D]
        """
        B, N, C = x.shape
        
        # Fused QKV: [B, N, 3*D] -> [B, N, 3, H, head_dim] -> [3, B, H, N, head_dim]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        
        # Efficient attention with causal mask
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True, 
        )
        
        # Reshape back: [B, H, N, head_dim] -> [B, N, D]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CausalTransformerBlock(TransformerBlock):
    """
    Causal version of stable-pretraining's TransformerBlock.
    
    Simply replaces the Attention module with CausalAttention.
    All other functionality (MLP, norms, AdaLN, etc.) inherited from parent.
    
    Args:
        Same as TransformerBlock, but self_attn is always True and
        cross_attn/use_adaln are forced to False for simplicity.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: type = nn.GELU,
    ):
        # Initialize parent with self_attn=True, cross_attn=False, use_adaln=False
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
            drop_path=drop_path,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            act_layer=act_layer,
        )
        
        # Replace self.attn with CausalAttention
        self.attn = CausalAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )




class CPCBackboneTransformer(nn.Module):
    """
    CPC Backbone with Causal Masked Self-Attention Transformer.
    
    Replaces GRU with a stack of causal transformer blocks for context modeling.
    Uses stable-pretraining's TransformerBlock with causal masking.
    
    Architecture:
        Input -> SincBlock -> Encoder -> Causal Transformer -> Context vectors
    
    Args:
        sample_rate (int): Audio sample rate (default: 3200 Hz for fin-whale)
        enc_hidden (int): Encoder output dimension (default: 512)
        transformer_hidden (int): Transformer hidden dimension (default: 256)
        sinc_channels (int): SincNet output channels (default: 512)
        num_layers (int): Number of transformer layers (default: 4)
        num_heads (int): Number of attention heads (default: 8)
        mlp_ratio (float): MLP hidden dimension ratio (default: 4.0)
        drop_path (float): Stochastic depth rate (default: 0.0)
        attn_drop (float): Attention dropout rate (default: 0.0)
        proj_drop (float): Projection dropout rate (default: 0.1)
        pos_encoding (str): Type of positional encoding ('sinusoidal' or 'learnable')
    """
    def __init__(
        self,
        sample_rate: int = 3200,
        enc_hidden: int = 512,
        transformer_hidden: int = 256,
        sinc_channels: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        pos_encoding: str = 'sinusoidal',
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.enc_hidden = enc_hidden
        self.transformer_hidden = transformer_hidden
        self.num_layers = num_layers
        
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
        self.encoder = CPCEncoder(in_chan=sinc_channels, enc_hidden=enc_hidden)
        
        # Project encoder output to transformer dimension
        self.input_proj = nn.Linear(enc_hidden, transformer_hidden)
        
        # Positional encoding (directly from stable-pretraining)
        if pos_encoding == 'learnable':
            # Learnable positional embeddings
            self.pos_embed = nn.Parameter(torch.zeros(1, 5000, transformer_hidden))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Fixed sinusoidal from stable-pretraining (returns torch.Tensor)
            pe = get_1d_sincos_pos_embed(transformer_hidden, 5000, cls_token=False)
            self.register_buffer('pos_embed', pe.unsqueeze(0))
        
        # Stack of causal transformer blocks (from stable-pretraining)
        # Use stochastic depth with linearly increasing drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.transformer_blocks = nn.ModuleList([
            CausalTransformerBlock(
                dim=transformer_hidden,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(transformer_hidden)
        
        # Dropout
        self.dropout = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass producing both encoded features and context vectors.
        
        Args:
            x: Input waveform [B, 1, T] or [B, T]
            
        Returns:
            c_t: Context vectors from Transformer [B, T', transformer_hidden]
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
        
        # Project to transformer dimension
        x_trans = self.input_proj(z)  # [B, T'', transformer_hidden]
        
        # Add positional encoding (directly from buffer/parameter)
        seq_len = x_trans.size(1)
        x_trans = x_trans + self.pos_embed[:, :seq_len, :]
        x_trans = self.dropout(x_trans)
        
        # Apply causal transformer blocks
        for block in self.transformer_blocks:
            x_trans = block(x_trans)
        
        # Final normalization
        c_t = self.final_norm(x_trans)  # [B, T'', transformer_hidden]
        
        return c_t, z
    
    def extract_embedding(self, x: torch.Tensor):
        """
        Extract fixed-size embedding by averaging context vectors.
        Used during validation/testing for downstream tasks.
        
        Args:
            x: Input waveform [B, 1, T] or [B, T]
            
        Returns:
            embedding: Fixed-size embedding [B, transformer_hidden], L2-normalized
        """
        with torch.no_grad():
            c_t, _ = self.forward(x)
            # Average pool over time
            embedding = c_t.mean(dim=1)  # [B, transformer_hidden]
            # L2 normalize for better downstream performance
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


def cpc_backbone_transformer(sample_rate=16000, **kwargs):
    """
    Factory function for CPC backbone with Transformer.
    
    Args:
        sample_rate (int): Audio sample rate
        **kwargs: Additional arguments for CPCBackboneTransformer
        
    Returns:
        CPCBackboneTransformer instance
    """
    return CPCBackboneTransformer(sample_rate=sample_rate, **kwargs)
