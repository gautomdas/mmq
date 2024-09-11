import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset 
import json
import re
import numpy as np

class VQAv2Eval(Dataset):
    def __init__(self, image_root, ann_root, q_root, img_transform=None, text_processor=None, prompt=None):
        self.annotation_dict = json.load(open(os.path.join(ann_root, "v2_mscoco_val2014_annotations.json")))
        print(len(self.annotation_dict["annotations"]))
        self.annotations = self.annotation_dict["annotations"]
        self.question_dict = json.load(open(os.path.join(q_root, "v2_OpenEnded_mscoco_val2014_questions.json")))
        self.questions = self.question_dict["questions"]
        self.image_root = image_root
        self.img_transform = img_transform
        self.text_processor = text_processor
        self.prompt = prompt if prompt else "Question: {} Short answer:"
        
        self.qa_pairs = []

        self._create_qa_pairs()
        """
        qa_pairs_path = os.path.join(ann_root, "qa_pairs.json")
        if os.path.isfile(os.path.join(ann_root, "qa_pairs.json")):
            self.qa_pairs = json.load(open(qa_pairs_path))
        else:
            self._create_qa_pairs()
            with open(qa_pairs_path, 'w') as file:
                json.dump(self.qa_pairs, file)
        """


    def set_max_samples(self, max_samples):
        self.qa_pairs = self.qa_pairs[:max_samples]
        question_ids = set([qa["question_id"] for qa in self.qa_pairs])

        self.annotations = list(filter(lambda anno: anno["question_id"] in question_ids, self.annotations))
        self.annotation_dict["annotations"] = self.annotations
        self.questions = list(filter(lambda question: question["question_id"] in question_ids, self.questions))
        self.question_dict["questions"] = self.questions

    def collater(self, samples):
        samples = [s for s in samples if s is not None]

        if not samples:
            return None

        images = []
        questions = []
        question_ids = []
        for sample in samples:
            images.append(sample["image"])
            questions.append(sample["text_input"])
            question_ids.append(sample["question_ids"])

        return {
            "image": images,
            "questions": questions,
            "question_ids": question_ids
        }


    def _create_qa_pairs(self):
        # image_id -> path map
        image_to_path = [entry.name for entry in os.scandir(self.image_root) if entry.is_file()]
        image_to_path = {int(re.search(r"_(\d+)\.jpg$", path).group(1)):path for path in image_to_path}

        #image_order = [anno["image_id"] for anno in self.annotations]
        #image_to_path = sorted(image_to_path, key=lambda image_id: image_order.index(image_id))

        question_to_annotation = {int(anno["question_id"]):anno for anno in self.annotations}
        #image_to_question = {int(question["image_id"]):question for question in self.questions["questions"]}

        for question in self.questions:
            image_id = question["image_id"]
            question_id = question["question_id"]
            annotation = question_to_annotation[question_id]
            self.qa_pairs.append({
                "question_id": question_id,
                "question": self.prompt.format(question["question"]),
                "answer": [answer_dict["answer"] for answer_dict in annotation["answers"]],
                "image": image_to_path[image_id]
            })
        """
        for image_id, image_path in image_to_path.items():
            annotation = image_to_annotation[image_id]
            question = image_to_question[image_id]
            self.qa_pairs.append({
                "question_id": question["question_id"],
                "question": self.prompt.format(question["question"]),
                "answer": [answer_dict["answer"] for answer_dict in annotation["answers"]],
                "image": image_path,
            })
        """

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.qa_pairs[index]

        image = Image.open(self.image_root+'/'+ann["image"]).convert("RGB")

        image = self.img_transform(image) if self.img_transform else image
        question = self.text_processor(ann["question"]) if self.text_processor else ann["question"]

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"]
        }
        """
        return {
            "inputs": inputs,
            "question_id": ann["question_id"]
        }
        """
        #    "instance_id": ann["instance_id"]

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