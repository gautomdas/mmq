from datasets import Flickr30kEvalDataset
import numpy as np
import re
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode 
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline
from transformers import Blip2ForImageTextRetrieval, Blip2Config

def process_caption(caption):
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

img_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

flickr30k = Flickr30kEvalDataset(
    "./data/flickr30k/images_flickr_1k_test", 
    "./data/flickr30k/annotations", 
    img_transform=img_transform,
    txt_processor=process_caption,
)

config = Blip2Config.from_pretrained("Salesforce/blip2-itm-vit-g")
model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16).to(device)
#print(config)

inferencer = InferencePipeline(model, device)
scorer = ScoringPipeline()

results = inferencer.run_inference(flickr30k, task="image_text_retrieval")
retrieval_results = scorer.compute_scores(results, "image_text_retrieval")

print(retrieval_results)