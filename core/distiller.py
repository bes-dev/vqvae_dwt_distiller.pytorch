import os
import random
# pytorch
import torch
import torch.nn as nn
import pytorch_lightning as pl
# models
from core.models import VQVAE, DecoderDWT
# loss
from core.loss import DistillerLoss
# dataset
from core.dataset import ImageDataset, FakeDataset
# configs
from omegaconf import OmegaConf
# utils
from core.utils import load_weights

class Distiller(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg

        # teacher model
        print("load teacher model...")
        vqvae_cfg = OmegaConf.load(cfg.teacher.data_path)
        self.vqvae, self.vqvae_preprocess = VQVAE.from_pretrained(vqvae_cfg)
        print("load student model...")
        self.student = DecoderDWT(
            vqvae_cfg.params.model.params.ddconfig,
            vqvae_cfg.params.model.params.embed_dim
        )
        print("copy weights for teacher to student...")
        load_weights(self.student.post_quant_conv, self.vqvae.model.post_quant_conv.state_dict())
        load_weights(self.student.decoder, self.vqvae.model.decoder.state_dict())
        # loss function
        self.loss = DistillerLoss(**self.cfg.loss)
        # dataset
        self.trainset = ImageDataset(transforms=self.vqvae_preprocess, **self.cfg.trainset)
        print(f"trainset size: {len(self.trainset)}")

    def _log_loss(self, loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, exclude=["loss"]):
        for k, v in loss.items():
            if not k in exclude:
                self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        if optimizer_idx == 0:
            loss = self.generator_step(batch, batch_nb)
        elif optimizer_idx == 1:
            loss = self.discriminator_step(batch, batch_nb)
        self._log_loss(loss)
        return {"loss": loss["loss"]}

    def generator_step(self, batch, batch_nb):
        embs = self.make_sample(batch)
        pred = self.student(embs)
        loss = self.loss.loss_g(pred, batch)
        return loss

    def discriminator_step(self, batch, batch_nb):
        embs = self.make_sample(batch)
        with torch.no_grad():
            pred = self.student(embs)
        if self.global_step % self.cfg.trainer.reg_d_interval != 0:
            loss = self.loss.loss_d(pred, batch)
        else:
            loss = self.loss.reg_d(batch)
            loss["loss"] *= self.cfg.trainer.reg_d_interval
        return loss

    def validation_step(self, batch, batch_nb):
        self._log_loss({"loss_val": 0})

    @torch.no_grad()
    def make_sample(self, batch):
        ids = self.vqvae.img_to_ids(batch["src"])
        embs = self.vqvae.ids_to_embs(ids)
        return embs

    def configure_optimizers(self):
        opts = []
        # self.student.decoder.conv_out ?
        opts.append(torch.optim.Adam(self.student.parameters(), lr=self.cfg.trainer.lr_student))
        # opts.append(torch.optim.Adam(self.student.decoder.conv_out.parameters(), lr=self.cfg.trainer.lr_student))
        opts.append(torch.optim.Adam(self.loss.gan_loss.parameters(), lr=self.cfg.trainer.lr_gan))
        return opts, []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.cfg.trainer.batch_size,
            num_workers=self.cfg.trainer.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            FakeDataset(),
            batch_size=1,
            num_workers=0,
            shuffle=False
        )
