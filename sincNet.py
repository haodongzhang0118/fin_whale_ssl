import torch
import torch.nn as nn
from sincConv import SincConv

def sinc_kernel_size(sample_rate):
    k = int(sample_rate // 126)
    if k % 2 == 0:
        k += 1
    return k

class PSwish(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        shape = (1, num_features, 1)
        self.p_swish_alpha = nn.Parameter(torch.empty(*shape))
        self.p_swish_beta  = nn.Parameter(torch.empty(*shape))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.p_swish_alpha, 2.0)
        nn.init.zeros_(self.p_swish_beta)

    def forward(self, x):
        # x: [B, C, T]
        return x * self.p_swish_alpha * torch.sigmoid(self.p_swish_beta * x)
    
class SincBlock(nn.Module):
    def __init__(self, out_channels=512, stride=1,
                 sample_rate=3200, return_abs=False,
                 learnable_filters=False, padding="same"):
        super().__init__()

        self.sinc = SincConv(
            out_channels=out_channels,
            kernel_size=251,         
            in_channels=1,
            stride=stride,                   
            dilation=1,
            padding=padding,                 
            padding_mode="reflect",
            sample_rate=sample_rate,
            min_low_hz=1,
            min_band_hz=10,
            learnable_filters=learnable_filters,
            apply_window_to_root=False,
            return_abs=return_abs,
            init_scale="mel",
        )
        self.act = PSwish(num_features=out_channels)

    def forward(self, x):   # x: [B,1,T]
        x = self.sinc(x)    # [B,512,T/5]
        x = self.act(x)
        return x
    
    
