"""
CPC for Fin-Whale Audio
Implementation based on stable-pretraining framework
"""

from .backbone import CPCBackbone, cpc_backbone, BRN1d
from .loss import CPCLoss, InfoNCELoss
from .forward import cpc_forward

__all__ = [
    'CPCBackbone',
    'cpc_backbone',
    'BRN1d',
    'CPCLoss',
    'InfoNCELoss',
    'cpc_forward',
]
