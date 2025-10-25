import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2ForImageTextRetrieval,
    Blip2Processor,
)

from blip_quantizer import BlipQuantizer
from dataset import COCODataset, Flickr30kEvalDataset
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline
from utils import compute_results, gather_results, init_distributed, load_quant_configs


def main(args):
    # Retrieval is not tested for distributed
    if args.task == "image_text_retrieval":
        args.distributed = False

    # Set up GPU/distributed environment
    if args.distributed:
        rank, world_size, gpu = init_distributed()
        dist.barrier()
    else:
        gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model, processor, dataset, an inference parameters
    inference_kwargs = {}
    if args.task == "blip2-image_captioning":
        model_name = "Salesforce/blip2-opt-2.7b"
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, device_map=gpu
        )
        processor = Blip2Processor.from_pretrained(model_name)
        dataset = COCODataset(
            ann_file=os.path.join(
                args.dataset_dir, "annotations/captions_val2017.json"
            ),
            img_dir=os.path.join(args.dataset_dir, "val2017"),
        )

    elif args.task == "blip2-image_text_retrieval":
        model_name = "Salesforce/blip2-itm-vit-g-coco"
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name, device_map=gpu)
        processor = Blip2Processor.from_pretrained(model_name)
        dataset = Flickr30kEvalDataset(
            ann_file=os.path.join(args.dataset_dir, "annotations/flickr30k_test.json"),
            img_dir=os.path.join(args.dataset_dir, "images_flickr_1k_test"),
        )
        if args.max_samples is not None:
            inference_kwargs["max_samples"] = args.max_samples
    else:
        raise ValueError(f"Unsupported task: {args.task}")

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
    # Load quantization config and quantize model
    config = load_quant_configs(args.config)
    quantizer = BlipQuantizer(model)
    quantizer.apply_quantization(config)

    # Task inference
    inferencer = InferencePipeline(model, gpu, processor)
    scorer = ScoringPipeline()

    results = inferencer.run_inference(
        dataloader,
        task=args.task,
        batch_size=args.batch_size,
        **inference_kwargs
    )

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
        description="Run task benchmarking with a given uniform quantization config"
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
        choices=["blip2-image_captioning", "blip2-image_text_retrieval"],
        help="Specify the task to run",
    )
    parser.add_argument(
        "--config", type=str, help="Path to the quantization config JSON file"
    )
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
