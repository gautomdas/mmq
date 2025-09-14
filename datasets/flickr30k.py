import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import re

class Flickr30kEvalDataset(Dataset):
    def __init__(self, ann_file, img_dir, img_transform=None):
        self.annotation = json.load(open(ann_file))
        self.img_transform = img_transform
        self.image_root = img_dir

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self._process_caption(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.annotation[index]["image"].split("/")[-1])
        image = Image.open(image_path).convert("RGB")
        if (self.img_transform):
            image = self.img_transform(image)

        return {"image": image, "index": index}

    def _process_caption(self, caption):
        max_words = 50
    
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
    
        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[: max_words])
    
        return caption
