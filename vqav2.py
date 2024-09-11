import json
import re

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import Blip2ForConditionalGeneration, Blip2Processor, T5TokenizerFast 

from datasets import VQAv2Eval
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline

multi_gpu = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up VQAv2 Dataset
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

vqav2 = VQAv2Eval(
    "./data/vqav2/val2014",
    "./data/vqav2/annotations",
    "./data/vqav2/questions",
)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", load_in_8bit=True, torch_dtype=torch.float16
)

# Set up DDP inference across gpus
if multi_gpu:
    num_workers = 4
    batch_size = 64

    t5_tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-xl")

    def process_question(question, max_words=50):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > max_words:
            question = " ".join(question_words[:max_words])

        text_tokens = t5_tokenizer(question, padding="longest", return_tensors="pt") 
        return text_tokens 


    img_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            ),
        ]
    )

    # Create DataLoader
    dataloader = DataLoader(
        vqav2, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True,
        shuffle=False,
        collate_fn=vqav2.collater,
    )

    n_gpus = torch.cuda.device_count()
    device_ids = list(range(n_gpus))

    model = DDP(model, device_ids)

inferencer = InferencePipeline(model, device, processor)
scorer = ScoringPipeline()

results = inferencer.run_inference(
    vqav2, task="visual_question_answering", max_samples=10000
)
vqa_results = scorer.compute_scores(results, "visual_question_answering")

print(vqa_results)

with open("./results.json", "w") as f:
    json.dump(vqa_results, f)
