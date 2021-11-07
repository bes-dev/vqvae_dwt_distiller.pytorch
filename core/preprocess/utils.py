import cv2
import torch
import numpy as np


def tensor_to_img(tensor, normalize=True, vrange=(-1, 1), to_numpy=True, rgb2bgr=True):
    if normalize:
        tensor = torch.clamp(tensor, min=vrange[0], max=vrange[1])
        tensor = (tensor - vrange[0]) / (vrange[1] - vrange[0] + 1e-5)
    img = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    if to_numpy:
        img = img.to('cpu', torch.uint8).numpy()
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def img_to_tensor(img, normalize=True, vrange=(0.0, 255.0), bgr2rgb=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    if normalize:
        tensor = torch.clamp(tensor, min=vrange[0], max=vrange[1])
        tensor = (tensor - vrange[0]) / (vrange[1] - vrange[0] + 1e-5)
    tensor = 2.0 * tensor - 1.0
    return tensor


def tokens_to_tensor(tokens, target_length, pad_id):
    pad_size = target_length - len(tokens)
    if pad_size > 0:
        tokens = np.hstack((tokens, np.full(pad_size, pad_id)))
    if len(tokens) > target_length:
        tokens = tokens[:target_length]
    return torch.tensor(tokens).long()
