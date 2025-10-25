import os

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def set_max_samples(self, max_samples):
        if max_samples > len(self.ids):
            return ValueError(
                f"Max_smaples: {max_samples}, is larger than the current size of the dataset {len(self.ids)}"
            )
        self.ids = self.ids[:max_samples]

    def collater(self, samples):
        samples = [s for s in samples if s is not None]

        if not samples:
            return None

        images = []
        captions = []
        image_ids = []
        for sample in samples:
            images.append(sample["image"])
            captions.append(sample["caption"])
            image_ids.append(sample["image_id"])

        return {"image": images, "caption": captions, "image_id": image_ids}

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann["caption"] for ann in anns]

        return {"image": image, "caption": captions, "image_id": img_id}

    def show_image(self, img_id):
        return self.coco.loadImgs(img_id)[0]

    def get_captions(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        return [ann["caption"] for ann in anns]
