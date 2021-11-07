import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from einops import rearrange
from .gumbel_vq import GumbelVQ
from core.preprocess.image_preprocess import OpenCVImagePreprocess


class VQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = GumbelVQ(
            ddconfig = cfg.model.params.ddconfig,
            n_embed = cfg.model.params.n_embed,
            embed_dim = cfg.model.params.embed_dim,
            kl_weight = cfg.model.params.kl_weight
        )
        self.num_layers = int(math.log(cfg.model.params.ddconfig.attn_resolutions[0]) / math.log(2))
        self.image_size = 256
        self.num_tokens = cfg.model.params.n_embed

    def img_to_ids(self, img):
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices, 'b h w -> b (h w)')

    def ids_to_embs(self, ids):
        b, n = ids.shape
        one_hot = F.one_hot(ids, num_classes=self.num_tokens).float()
        embs = (one_hot @ self.model.quantize.embed.weight)
        embs = rearrange(embs, 'b (h w) c -> b c h w', h = int(math.sqrt(n)))
        return embs

    def embs_to_img(self, embs):
        img = self.model.decode(embs)
        return img

    @classmethod
    def from_pretrained(cls, cfg):
        print(f"[{cls.__name__}]: create model")
        model = cls(cfg.params)
        print(f"[{cls.__name__}]: load checkpoint {cfg.ckpt}")
        ckpt = torch.load(hf_hub_download(**cfg.ckpt), map_location="cpu")["state_dict"]
        model.model.load_state_dict(ckpt, strict=False)
        print(f"[{cls.__name__}]: create preprocessor")
        model_preprocess = OpenCVImagePreprocess()
        return model, model_preprocess
