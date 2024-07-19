import argparse
import os
from blip_quantizer import BlipQuantizer, QuantConfig, ModelPart, LayerGroup, LayerType
from quant_functions import uniform_quantization
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import COCODataset
from inference_pipeline import InferencePipeline
from utils import load_quant_configs

def main(config_file):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and processor
    model_name = "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    processor = Blip2Processor.from_pretrained(model_name)
    
    # Load quantization configs
    configs = load_quant_configs(config_file)
    
    # Quantize model
    print("Quantizing model...")
    quantizer = BlipQuantizer(model)
    quantizer.apply_quantization(configs)

    print(f"Quantized {quantizer.count_quantized_layers()} layers at {quantizer.get_bits()} bits")
    
    # Set up dataset
    coco_dataset = COCODataset(ann_file='../datasets/cocow/annotations/captions_val2017.json',
                               img_dir='../datasets/cocow/images/val2017')
    
    # Run inference
    inferencer = InferencePipeline(model, processor, device)
    print("Starting inference...")
    results = inferencer.run_inference(coco_dataset, task='image_captioning', max_samples=1000)
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    result_file = os.path.join('./results', os.path.basename(config_file))
    print(f"Inference Finished, Saving Results to {result_file}...")
    inferencer.save_results(results, result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a given quantization config")
    parser.add_argument("config", help="Path to the quantization config JSON file")
    args = parser.parse_args()
    
    main(args.config)
