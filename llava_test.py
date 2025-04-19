import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
# TODO remove
from transformers.models.clip.modeling_clip import CLIPVisionTransformer as cvpt

from dataset import VQAv2Eval
from inference_pipeline import InferencePipeline
import time
from scoring_pipeline import ScoringPipeline

from dataset import VQAv2Eval
import os
from awq.llava_quantizer import LlavaAWQQuantizer


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

# VQAv2 dataset paths
ann_root = '/fs/cfar-projects/low-bit-vision/datasets/vqav2/annotations'
q_root = '/fs/cfar-projects/low-bit-vision/datasets/vqav2/questions'
image_root = '/fs/cfar-projects/low-bit-vision/datasets/vqav2/val2014'


dataset = VQAv2Eval(image_root=image_root,
                    ann_root=ann_root,
                    q_root=q_root,)

# Load the model
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", pad_token = '<pad>')

config = {}

# TODO:
# config['vision_layers'] = {}

config['llm_layers'] = {
    'self_attn': 4,
    'mlp': 4
}

quantizer = LlavaAWQQuantizer(model, device, processor, dataset, config)

quantizer.quantize()
