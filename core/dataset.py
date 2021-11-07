import os
import torch
import cv2

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, size_src, size_tgt, transforms=None):
        super().__init__()
        self.data_path = data_path
        self.files = open(os.path.join(data_path, "files.txt")).readlines()
        self.size_src = size_src
        self.size_tgt = size_tgt
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        out = {}
        img = cv2.imread(os.path.join(self.data_path, self.files[idx].strip()))
        out["src"] = self.transforms.encode(img, (self.size_src, self.size_src)).squeeze(0)
        out["tgt"] = self.transforms.encode(img, (self.size_tgt, self.size_tgt)).squeeze(0)
        return out

class FakeDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.zeros(1)
