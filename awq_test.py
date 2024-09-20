import torch
import numpy as np
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, Blip2ForImageTextRetrieval
from datasets import COCODataset
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
# from utils import print_model_structure
from inference_pipeline import InferencePipeline


# from collections import defaultdict
# from functools import partial

# import gc
# import inspect
# import random
from typing import Tuple, List

from awq.quantizer import Blip2ForConditionalGenerationAWQQuantizer

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "Salesforce/blip2-opt-2.7b"
model = Blip2ForConditionalGeneration.from_pretrained(model_name)
# model = model.cpu()
model.to(device)

processor = Blip2Processor.from_pretrained(model_name)

# on NEXUS
coco_dataset = COCODataset(ann_file='/nfshomes/vla/project_dirs/low-bit-vision/datasets/cocow/annotations/captions_val2017.json',
                           img_dir='/nfshomes/vla/project_dirs/low-bit-vision/datasets/cocow/images/val2017')

b = Blip2ForConditionalGenerationAWQQuantizer(model, device, processor, coco_dataset)
layers, first_inputs, layer_args, layer_kwargs, linear_inputs, scales = b.quantize()
