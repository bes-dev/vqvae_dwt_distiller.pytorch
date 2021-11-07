import torch
import torch.nn as nn
import torch.nn.functional as F
from core.loss.non_saturating_gan_loss import NonSaturatingGANLoss
from core.loss.perceptual_loss import PerceptualLoss
from pytorch_wavelets import DWTInverse, DWTForward


class DistillerLoss(nn.Module):
    def __init__(
            self,
            discriminator_size=512,
            perceptual_size=256,
            loss_weights={"l1": 1.0, "l2": 1.0, "loss_p": 1.0, "loss_g": 0.5}
    ):
        super().__init__()
        self.loss_weights = loss_weights
        # l1/l2 loss
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        # perceptual_loss
        self.perceptual_loss = PerceptualLoss(perceptual_size)
        # gan loss
        self.gan_loss = NonSaturatingGANLoss(image_size=int(discriminator_size))
        # utils
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        self.idwt = DWTInverse(mode="zero", wave="db1")

    def loss_g(self, pred, gt):
        # prepare gt
        tgt = gt["tgt"]
        tgt_freq = self.img_to_dwt(tgt)
        # compute pixelwise loss
        loss = {}
        loss["l1"] = self.l1_loss(pred["img"], tgt) + self.l1_loss(pred["freq"], tgt_freq)
        loss["l2"] = self.l2_loss(pred["img"], tgt) + self.l2_loss(pred["freq"], tgt_freq)
        # perceptual_loss
        loss["loss_p"] = self.perceptual_loss(pred["img"], tgt)
        # gan loss
        loss["loss_g"] = self.gan_loss.loss_g(pred["img"], tgt)

        # total loss
        loss["loss"] = 0
        for k, w in self.loss_weights.items():
            if loss[k] is not None:
                loss["loss"] += w * loss[k]
            else:
                del loss[k]
        return loss

    def loss_d(self, pred, gt):
        loss = {}
        loss["loss"] = loss["loss_d"] = self.gan_loss.loss_d(pred["img"].detach(), gt["tgt"])
        return loss

    def reg_d(self, real):
        out = {}
        out["loss"] = out["d_reg"] = self.gan_loss.reg_d(real["tgt"])
        return out

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq
