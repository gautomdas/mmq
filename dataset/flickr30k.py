import json
import os
import re

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Flickr30kEvalDataset(Dataset):
    def __init__(self, ann_file, img_dir, img_transform=None):
        self.annotation = json.load(open(ann_file))
        self.img_transform = img_transform
        self.image_root = img_dir

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        if img_transform is None:
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize((364, 364), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

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

    def set_max_samples(self, max_samples):
        if max_samples > len(self.annotation):
            return ValueError(
                f"Max_samples: {max_samples}, is larger than the current size of the dataset{len(self.annotation)}"
            )
        self.annotation = self.annotation[:max_samples]

    def __getitem__(self, index):
        image_path = os.path.join(
            self.image_root, self.annotation[index]["image"].split("/")[-1]
        )
        image = Image.open(image_path).convert("RGB")
        if self.img_transform:
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
            caption = " ".join(caption_words[:max_words])

        return caption
