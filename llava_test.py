import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from dataset import VQAv2Eval
# from inference_pipeline import InferencePipeline
import time
# from scoring_pipeline import ScoringPipeline

from dataset import VQAv2Eval
# import os
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

# config['vision_layers'] = {
#     'self_attn':4,
#     'mlp': 4
# }

config['llm_layers'] = {
    'self_attn': 4,
    'mlp': 4
}

quantizer = LlavaAWQQuantizer(model, device, processor, dataset, config)
quantizer.n_samples = 64

start_time = time.time()
quantizer.quantize()
elapsed_time = time.time() - start_time

print(f'Elapsed time: {elapsed_time} seconds')

img = dataset[42]['image']
prompt = 'USER: <image>\n' + dataset.qa_pairs[42]['question'] + '\nAnswer the question using a single word or phrase. ASSISTANT:'

model = model.to('cuda')
samples = processor(images = [img],
                     text=[prompt],
                     return_tensors='pt',
                     padding=True).to(model.device)

generate_ids = model.generate(**samples)
print(processor.batch_decode(generate_ids, skip_special_tokens=True))
