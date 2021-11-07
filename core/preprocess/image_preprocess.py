import cv2
from .utils import img_to_tensor, tensor_to_img


class OpenCVImagePreprocess:
    @staticmethod
    def encode(img, size=None, normalize=True, vrange=(0.0, 255.0), bgr2rgb=True):
        if size is not None:
            img = cv2.resize(img, size)
        tensor = img_to_tensor(img, normalize, vrange, bgr2rgb)
        return tensor

    @staticmethod
    def decode(tensor, normalize=True, vrange=(-1, 1), to_numpy=True, rgb2bgr=True):
        imgs = []
        for i in range(tensor.size(0)):
            img = tensor_to_img(tensor[i], normalize, vrange, to_numpy, rgb2bgr)
            imgs.append(img)
        return imgs
