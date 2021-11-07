import torch
import torch.nn as nn
from taming.modules.diffusionmodules.model import Decoder
from .idwt import DWTInverse


class DecoderDWT(nn.Module):
    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        if ddconfig.out_ch != 12:
            ddconfig.out_ch = 12
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)
        self.decoder = Decoder(**ddconfig)
        self.idwt = DWTInverse(mode="zero", wave="db1")

    def forward(self, x):
        x = self.post_quant_conv(x)
        freq = self.decoder(x)
        img = self.dwt_to_img(freq)
        return {"freq": freq, "img": img}

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))
