import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class VQA(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, delimiter="|", skipinitialspace=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Group captions by image
        self.grouped_data = self.data.groupby('image_name')
        self.image_names = list(self.grouped_data.groups.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        captions = list(self.grouped_data.get_group(img_name)['comment'])
        
        return image, captions