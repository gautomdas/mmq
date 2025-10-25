import argparse
import builtins as __builtin__
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from dataset import GQAEval
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline


def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(gpu)

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    return rank, world_size, gpu


def compute_gqa_results(results, scorer, save_path=None):
    gqa_results = scorer.compute_scores(results, "gqa")
    print(gqa_results)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(gqa_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GQA Balanced-Testdev Eval",
        description="Performs VQA evaluation using BLIP2 on GQA",
    )

    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--dataset_dir", default="./data/gqa", type=str)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = Blip2Processor.from_pretrained(
        "salesforce/blip2-opt-2.7b", padding_side="left"
    )
    gqa = GQAEval(
        os.path.join(args.dataset_dir, "images"),
        os.path.join(args.dataset_dir, "questions"),
    )

    if args.distributed:
        rank, world_size, gpu = init_distributed()
        dist.barrier()

        try:
            sampler = DistributedSampler(
                gqa, shuffle=False, num_replicas=world_size, rank=rank
            )

            dataloader = DataLoader(
                gqa,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=False,
                shuffle=False,
                sampler=sampler,
                collate_fn=gqa.collater,
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", device_map=gpu
            )

            inferencer = InferencePipeline(model, gpu, processor)
            scorer = ScoringPipeline()

            # T5 kwargs
            # processor_kwargs={"padding": "longest", "max_length": 32, "truncation": True}
            # generate_kwargs={"num_beams": 5, "max_new_tokens": 10, "min_length": 1, "length_penalty": -1, "do_sample": False}
            # OPT kwargs
            processor_kwargs = {
                "padding": "longest",
                "max_length": 32,
                "truncation": True,
            }
            generate_kwargs = {
                "num_beams": 5,
                "max_new_tokens": 10,
                "min_length": 1,
                "length_penalty": 0,
                "do_sample": False,
            }

            results = inferencer.run_inference(
                dataloader,
                task="gqa",
                processor_kwargs=processor_kwargs,
                generate_kwargs=generate_kwargs,
            )

            with open(os.path.join(args.output_dir, f"{rank}_results.json"), "w") as f:
                json.dump(results, f)
            dist.barrier()

            if rank == 0:
                results = []

                question_ids = set()
                for rank_id in range(world_size):
                    with open(
                        os.path.join(args.output_dir, f"{rank_id}_results.json"), "r"
                    ) as f:
                        rank_results = json.load(f)
                        for answer in rank_results:
                            question_id = answer["question_id"]
                            if question_id not in question_ids:
                                results.append(answer)
                                question_ids.add(question_id)

                compute_gqa_results(
                    results, scorer, os.path.join(args.output_dir, "results.json")
                )
        finally:
            dist.destroy_process_group()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", device_map=device
        )

        inferencer = InferencePipeline(model, device, processor)
        scorer = ScoringPipeline()

        # T5 kwargs
        # processor_kwargs={"padding": "longest", "max_length": 32, "truncation": True}
        # generate_kwargs={"num_beams": 5, "max_new_tokens": 10, "min_length": 1, "length_penalty": -1, "do_sample": False}
        # OPT kwargs
        processor_kwargs = {"padding": "longest", "max_length": 32, "truncation": True}
        generate_kwargs = {
            "num_beams": 5,
            "max_new_tokens": 10,
            "min_length": 1,
            "length_penalty": 0,
            "do_sample": False,
        }

        dataloader = DataLoader(
            gqa,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle=False,
            collate_fn=gqa.collater,
        )

        results = inferencer.run_inference(
            dataloader,
            task="gqa",
            processor_kwargs=processor_kwargs,
            generate_kwargs=generate_kwargs,
        )

        with open(os.path.join(args.output_dir, "answers.json"), "w") as f:
            json.dump(results, f)

        compute_gqa_results(
            results, scorer, os.path.join(args.output_dir, "results.json")
        )
