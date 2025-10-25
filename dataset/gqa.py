import json
import os

from PIL import Image
from torch.utils.data import Dataset


class GQAEval(Dataset):
    """
    image_root (string): Root directory of images
    q_root (string): Root directory of questions
    """

    def __init__(
        self, image_root, q_root, img_transform=None, text_processor=None, prompt=None
    ):
        self.image_root = image_root
        with open(os.path.join(q_root, "testdev_balanced_questions.json"), "r") as f:
            self.questions = json.load(f)
        self.question_ids = list(self.questions.keys())
        self.img_transform = img_transform
        self.text_processor = text_processor
        self.prompt = prompt if prompt else "Question: {} Short answer:"

    def __len__(self):
        return len(self.question_ids)

    def set_max_samples(self, max_samples):
        if max_samples > len(self.question_ids):
            return ValueError(
                f"Max_samples: {max_samples}, is larger than the current size of the dataset {len(self.question_ids)}"
            )
        self.question_ids = self.question_ids[:max_samples]

    def collater(self, samples):
        samples = [s for s in samples if s is not None]

        if not samples:
            return None

        question_ids = []
        images = []
        questions = []
        gt_answers = []
        for sample in samples:
            question_ids.append(sample["question_id"])
            images.append(sample["image"])
            questions.append(sample["text_input"])
            gt_answers.append(sample["gt_answer"])

        return {
            "question_id": question_ids,
            "image": images,
            "text_input": questions,
            "gt_answer": gt_answers,
        }

    def __getitem__(self, index):
        question_id = self.question_ids[index]
        question = self.questions[question_id]

        image_path = os.path.join(self.image_root, f"{question['imageId']}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)

        if self.text_processor:
            question = self.text_processor(question)

        return {
            "question_id": question_id,
            "image": image,
            "text_input": self.prompt.format(question["question"]),
            "gt_answer": question["answer"],
        }
