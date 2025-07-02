import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class CommonTestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)  # (C, H, W)
        return image.unsqueeze(0), img_path
