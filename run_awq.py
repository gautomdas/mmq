import argparse
import json
import os

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

from awq.llava_quantizer import LlavaAWQQuantizer
from awq.quantizer import (
    Blip2ForConditionalGenerationAWQQuantizer,
    Blip2ForImageTextRetrievalAWQQuantizer,
)
from dataset import COCODataset, Flickr30kEvalDataset, GQAEval, VQAv2Eval
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline
from utils import compute_results, gather_results, init_distributed

def main(args):
    # Retrieval is not tested for distributed
    if args.task == "blip2-image_text_retrieval":
        args.distributed = False

    # Set up GPU/distributed environment
    if args.distributed:
        rank, world_size, gpu = init_distributed()
        dist.barrier()
    else:
        gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = json.load(open(args.config))

    # Set up dataset
    llava_prompt = "USER: <image>\n{}\nAnswer the question using a single word or phrase. ASSISTANT:"
    if "image_captioning" in args.task:
        dataset = COCODataset(
            ann_file=os.path.join(
                args.dataset_dir, "annotations/captions_val2017.json"
            ),
            img_dir=os.path.join(args.dataset_dir, "val2017"),
        )
    elif "image_text_retrieval" in args.task:
        # NOTE: img_transform set to None so that AWQ can use the Blip2Processor for calibration set
        dataset = Flickr30kEvalDataset(
            ann_file=os.path.join(args.dataset_dir, "annotations/flickr30k_test.json"),
            img_dir=os.path.join(args.dataset_dir, "images_flickr_1k_test"),
            img_transform=None,
        )
        dataset.img_transform = None
    elif "vqav2" in args.task:
        dataset = VQAv2Eval(
            image_root=os.path.join(args.dataset_dir, "val2014"),
            ann_root=os.path.join(args.dataset_dir, "annotations"),
            q_root=os.path.join(args.dataset_dir, "questions"),
            prompt=llava_prompt if "llava" in args.task else None,
        )
    elif "gqa" in args.task:
        dataset = GQAEval(
            image_root=os.path.join(args.dataset_dir, "images"),
            q_root=os.path.join(args.dataset_dir, "questions"),
            prompt=llava_prompt if "llava" in args.task else None,
        )
    else:
        raise ValueError("Unsupported task")

    # model specific transformers imports
    if "blip2" in args.task:
        from transformers import (
            Blip2ForConditionalGeneration,
            Blip2ForImageTextRetrieval,
            Blip2Processor
        )
    elif "llava" in args.task:
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
            LlavaImageProcessor
        )
    else:
        raise ValueError(f"Unknown model in task: {args.task}")


    # Set up model, processor, and quantizer
    if args.task in ("blip2-image_captioning", "blip2-vqav2", "blip2-gqa"):
        model_name = "Salesforce/blip2-opt-2.7b"
        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        processor = Blip2Processor.from_pretrained(model_name, padding_side="left")
        quantizer = Blip2ForConditionalGenerationAWQQuantizer(
            model, gpu, processor, dataset, config
        )
    elif args.task == "blip2-image_text_retrieval":
        model_name = "Salesforce/blip2-itm-vit-g-coco"
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        processor = Blip2Processor.from_pretrained(model_name)
        quantizer = Blip2ForImageTextRetrievalAWQQuantizer(
            model, gpu, processor, dataset, config
        )
    #elif args.task == "llava-vqav2" or args.task == "llava-gqa":
    elif args.task in ("llava-vqav2", "llava-gqa"):
        model_name = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(model_name).to(gpu)
        image_processor = LlavaImageProcessor.from_pretrained(model_name, do_pad=True)
        processor = AutoProcessor.from_pretrained(
            model_name, pad_token="<pad>", use_fast=False
        )
        processor.image_processor = image_processor
        processor.tokenizer.padding_side = "left"
        quantizer = LlavaAWQQuantizer(model, gpu, processor, dataset, config)
    else:
        raise ValueError("Unsupported task")

    # Set up inference parameters
    inference_kwargs = {}
    if args.task == "blip2-image_text_retrieval":
        if args.max_samples is not None:
            inference_kwargs["max_samples"] = args.max_samples
    elif args.task in ("blip2-vqav2", "blip2-gqa"):
        inference_kwargs["processor_kwargs"] = {
            "padding": "longest",
            "max_length": 32,
            "truncation": True,
        }
        inference_kwargs["generate_kwargs"] = {
            "num_beams": 5,
            "max_new_tokens": 10,
            "min_length": 1,
            "do_sample": False,
        }
        if model_name.startswith("Salesforce/blip2-opt"):
            inference_kwargs["generate_kwargs"]["length_penalty"] = 0
        elif model_name.startswith("Salesforce/blip2-flan-t5"):
            inference_kwargs["generate_kwargs"]["length_penaty"] = -1
    elif args.task in ("llava-vqav2", "llava-gqa"):
        inference_kwargs["processor_kwargs"] = {"padding": True}
        inference_kwargs["generate_kwargs"] = {"num_beams": 1, "do_sample": False}

    # Quantize the model
    quantizer.quantize()

    if args.task == "blip2-image_text_retrieval":
        dataset.img_transform = transforms.Compose(
            [
                transforms.Resize(
                    (364, 364), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    model = model.to(gpu)

    # Trim the dataset to the first n samples if max_samples is specified
    if args.max_samples is not None:
        dataset.set_max_samples(args.max_samples)

    # Set up dataloader
    if "image_text_retrieval" in args.task:
        dataloader = dataset
    elif args.distributed:
        sampler = DistributedSampler(
            dataset, shuffle=False, num_relicas=world_size, rank=rank
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False if gpu == "cpu" else True,
            shuffle=False,
            sampler=sampler,
            collate_fn=dataset.collater,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            # pin_memory=False if gpu == "cpu" else True,
            pin_memory=False,
            shuffle=False,
            collate_fn=dataset.collater,
        )

    # Task inference
    inferencer = InferencePipeline(model, gpu, processor)
    scorer = ScoringPipeline()

    results = inferencer.run_inference(
        dataloader,
        task=args.task,
        batch_size=args.batch_size,
        **inference_kwargs
    )
    if "vqav2" in args.task:
        results = {
            "answers": results,
            "annotations": os.path.join(
                args.dataset_dir, "annotations/v2_mscoco_val2014_annotations.json"
            ),
            "questions": os.path.join(
                args.dataset_dir, "questions/v2_OpenEnded_mscoco_val2014_questions.json"
            ),
        }

    # Task scoring
    if args.distributed:
        with open(os.path.join(args.output_dir, f"{rank}_results.json"), "w") as f:
            json.dump(results, f)
        dist.barrier()

        if rank == 0:
            results = gather_results(args, rank, world_size)

            compute_results(
                results,
                scorer,
                args.task,
                os.path.join(args.output_dir, "results.json"),
            )

        dist.destroy_process_group()
    else:
        compute_results(
            results, scorer, args.task, os.path.join(args.output_dir, "results.json")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a given quantization config"
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether to use distributed inference in a single node (default: False); only supported for image captioning and VQA tasks",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of threads for dataset loading (default: 1)",
    )
    parser.add_argument(
        "--task",
        choices=[
            "blip2-image_captioning",
            "blip2-image_text_retrieval",
            "blip2-vqav2",
            "blip2-gqa",
            "llava-vqav2",
            "llava-gqa",
        ],
    )
    parser.add_argument("--config", help="Path to the quantization config JSON file")
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="If specified, uses the first n samples from the dataset",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./data",
        type=str,
        help="Path to the dataset directory (default: ./data)",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output directory (default: ./output)",
    )

    args = parser.parse_args()

    main(args)
