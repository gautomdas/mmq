from awq.quantizer import Blip2ForConditionalGenerationAWQQuantizer, Blip2ForImageTextRetrievalAWQQuantizer
from dataset import COCODataset, Flickr30kEvalDataset
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline

import torch
import torchvision.transforms as transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, Blip2ForImageTextRetrieval

import numpy as np
import json
import os
import argparse


def main(config_path, task):

    config = json.load(open(config_path))

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model,dataset,quantizer based on task
    if task == 'image_captioning':
        model_name = "Salesforce/blip2-opt-2.7b"
        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        processor = Blip2Processor.from_pretrained(model_name)
        dataset = COCODataset(ann_file='/fs/cfar-projects/low-bit-vision/datasets/cocow/annotations/captions_val2017.json',
                              img_dir='/fs/cfar-projects/low-bit-vision/datasets/cocow/images/val2017')

        quantizer = Blip2ForConditionalGenerationAWQQuantizer(model, device, processor, dataset, config)


    elif task == 'image_text_retrieval':
        model_name = "Salesforce/blip2-itm-vit-g-coco"
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        processor = Blip2Processor.from_pretrained(model_name)

        # NOTE: img_transform set to None so that AWQ can use the Blip2Processor for calibration set
        dataset = Flickr30kEvalDataset(ann_file='/fs/cfar-projects/low-bit-vision/datasets/flickr30k/annotations/test.json', 
                                       img_dir = '/fs/cfar-projects/low-bit-vision/datasets/flickr30k/images/flickr30k-images', 
                                       img_transform=None)

        quantizer = Blip2ForImageTextRetrievalAWQQuantizer(model, device, processor, dataset, config)

        img_transform = transforms.Compose(
            [
                transforms.Resize(
                    (364, 364), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )

    # APPLY AWQ
    print('Applying AWQ Quantization...')
    quantizer.quantize()

    # need to set this for the inference pipeline
    if task == 'image_text_retrieval':
        dataset.img_transform = img_transform


    model = model.to(device)
    # RUN INFERENCE
    print(f'Running inference on {task} task...')
    pipeline = InferencePipeline(model, device, processor)
    results = pipeline.run_inference(dataset, task = task)
    
    os.makedirs(f'{task}_results', exist_ok=True)
    result_path = os.path.join(f"{task}_results", os.path.basename(config_path))
    print(f"Inference Finished, Saving Results to {result_path}...")

    results['model_size'] = quantizer.model_size

    if task == 'image_text_retrieval':
        for key in results:
            # cast these to lists so we can jsonify properly
            if type(results[key]) == np.ndarray:
                results[key] = results[key].tolist()

    pipeline.save_results(results, result_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run inference with a given quantization config"
    )
    parser.add_argument("--config_path", help="Path to the quantization config JSON file")
    parser.add_argument("--task", choices = ['image_captioning', 'image_text_retrieval'])

    args = parser.parse_args()

    main(args.config_path, args.task)