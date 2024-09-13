import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json

class Flickr30kEvalDataset(Dataset):
    def __init__(self, image_root, ann_root, img_transform=None, txt_processor=None):
        self.annotation = json.load(open(os.path.join(ann_root, "flickr30k_test.json")))
        self.img_transform = img_transform
        self.txt_processor = txt_processor 
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.txt_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.annotation[index]["image"].split("/")[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.img_transform(image)

        return {"image": image, "index": index}