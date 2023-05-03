import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

class ImageDataset(Dataset):
    def __init__(self, image_data, target_shape, transform=None):
        self.input_image_data = image_data['image']
        self.transform = transform
        self.target_shape = target_shape

    def __len__(self):
        return len(self.input_image_data)

    def _normalization(self, data):
        return data/255
        
    def __getitem__(self, idx):
        input = Image.open(self.input_image_data[idx]).convert("RGB")
        input = np.array(input)
        input_lr = cv2.resize(input, (self.target_shape, self.target_shape), interpolation=cv2.INTER_LINEAR)
        if self.transform:
            trasformed = self.transform(image=input, image1=input_lr)
            return trasformed['image'], trasformed['image1']
        else:
            input = self._normalization(torch.from_numpy(input).permute(2, 0, 1))
            target = self._normalization(torch.from_numpy(input).permute(2, 0, 1))
            return torch.FloatTensor(input), torch.FloatTensor(target)
