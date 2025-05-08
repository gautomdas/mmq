import sys
sys.path.append('..')
# import math
# import time
# from typing import List, Dict, Any, Optional
import argparse
import random
import os
import json

import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
# from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.models.llava.image_processing_llava import LlavaImageProcessor
# import transformers

from dataset import VQAv2Eval, GQAEval
from awq.llava_quantizer import LlavaAWQQuantizer
from inference_pipeline import InferencePipeline
# from scoring_pipeline import ScoringPipeline


def get_args():

    parser = argparse.ArgumentParser(description="LLAVA AWQ Quantization Script")

    parser.add_argument(
        '--task',
        type=str,
        choices=['vqav2', 'gqa'],
        required=True,
        help='task to evaluate AWQ-quantized LLAVA on'
    )

    # Add arguments for bit sizes
    parser.add_argument(
        "--vision-bits",
        type=int,
        default=8,
        choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help="Bit size for vision component",
    )

    parser.add_argument(
        "--language-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help="Bit size for language component",
    )

    parser.add_argument(
        "--calibration-size", type=int, default=128, help="Size of calibration dataset"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not provided, a random seed will be generated.",
    )


    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu, etc.)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="awq_results",
        help="Directory to save results",
    )

    parser.add_argument(
        '--no_quant',
        default=False,
        action='store_true',
        help="Set to true to apply no quantization (full-precision run)"
    )

    parser.add_argument(
        '--batch_size',
        type = int,
        default = 16,
        help = 'batch size for task evaulation'
    )

    return parser.parse_args()

def main():
    args = get_args()

     # Generate random seed if not provided
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {args.seed}")


    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device(
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )

    print("Loading LLAVA model...")
    # Load the model
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16)
    model.to(device)

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", pad_token = '<pad>', use_fast = False)

    # need to use this image processor w/ do_pad=True according to "Note regarding reproducing original implementation"
    # https://huggingface.co/docs/transformers/en/model_doc/llava
    image_processor = LlavaImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                           do_pad=True)

    processor.image_processor = image_processor

    # short answer prompting according to: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
    llava_prompt = 'USER: <image>\n{}\nAnswer the question using a single word or phrase. ASSISTANT:'
    
    if args.task == 'vqav2':
        # VQAv2 dataset paths
        ann_root = '/fs/cfar-projects/low-bit-vision/datasets/vqav2/annotations'
        q_root = '/fs/cfar-projects/low-bit-vision/datasets/vqav2/questions'
        image_root = '/fs/cfar-projects/low-bit-vision/datasets/vqav2/val2014'

        dataset = VQAv2Eval(image_root=image_root,
                            ann_root=ann_root,
                            q_root=q_root,
                            prompt = llava_prompt)
        
    elif args.task == 'gqa':
        # GQA dataset paths
        image_root = '/fs/cfar-projects/low-bit-vision/datasets/gqa/images'
        q_root = '/fs/cfar-projects/low-bit-vision/datasets/gqa/questions'

        dataset = GQAEval(
                image_root,
                q_root,
                prompt=llava_prompt
        )
    

    if not args.no_quant:
        # Update quantizer config with specified bit sizes
        config = {}

        config['vision_layers'] = {
            'self_attn': args.vision_bits,
            'mlp': args.vision_bits
        }

        config['llm_layers'] = {
            'self_attn': args.vision_bits,
            'mlp': args.vision_bits
        }

        # Print configuration
        print("\nQuantization Configuration:")
        print(f"Vision bits: {args.vision_bits}")
        print(f"Language bits: {args.language_bits}")
        print(f"Calibration size: {args.calibration_size}")
        print(f"Device: {device}\n")


        quantizer = LlavaAWQQuantizer(model, device, processor, dataset, config)
        quantizer.n_samples = args.calibration_size

        # Quantize model
        quantizer.quantize()

        model.to(device)

    # Evaluate on task
    gpu_name = torch.cuda.get_device_name()
    
    # adjust batch sizes depending on available gpu memory
    if "A5000" in gpu_name.replace(" ", ""):
        args.batch_size = 16
    elif "A6000" in gpu_name.replace(" ", ""):
        args.batch_size = 56

    print(f'Evaluating on task: {args.task}')
    print(f'batch_size: {args.batch_size}')
    quantizer.prepare_for_inference()

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=1,
                            pin_memory=False,
                            shuffle=False,
                            collate_fn = dataset.collater)
    
   
    inferencer = InferencePipeline(model, device, processor)

    # set this according to huggingface usage tips: https://huggingface.co/docs/transformers/en/model_doc/llava
    processor.tokenizer.padding_side = "left"
    processor_kwargs = dict(padding=True)

    # greedy decoding
    generate_kwargs = {
        'num_beams': 1,
        'do_sample': False
    }

    results = inferencer.run_inference(
        dataloader,
        task = args.task,
        processor_kwargs = processor_kwargs,
        generate_kwargs = generate_kwargs
    )


    json_out = {
        "answers": results,
        "vision_bits": args.vision_bits,
        "language_bits": args.language_bits
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"results_v{args.vision_bits}_l{args.language_bits}.json")
    with open(json_path, 'w') as f:
        json.dump(json_out, f)

    
    print(f"Output results to {json_path}")


if __name__ == '__main__':
    main()
