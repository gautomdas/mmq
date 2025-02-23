import argparse
import os
import json
import builtins as __builtin__

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from transformers import Blip2ForConditionalGeneration, Blip2Processor 

from datasets import VQAv2Eval
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline

def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu)

    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        if rank == 0:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

    return rank, world_size, gpu

def compute_vqa_results(results, scorer, save_path=None):
    vqa_results = scorer.compute_scores(results, "visual_question_answering")
    print(vqa_results)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(vqa_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='VQAv2 Eval',
        description='Performs VQA evaluation using BLIP2 on VQAv2',
    )

    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--output_dir", default="./output", type=str)

    args = parser.parse_args()

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    vqav2 = VQAv2Eval(
        "./data/vqav2/val2014",
        "./data/vqav2/annotations",
        "./data/vqav2/questions",
    )

    if args.distributed:
        rank, world_size, gpu = init_distributed()
        dist.barrier()

        sampler = DistributedSampler(
            vqav2,
            shuffle=False,
            num_replicas=world_size,
            rank=rank
        )
        
        dataloader = DataLoader(
            vqav2, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            pin_memory=False,
            shuffle=False,
            sampler=sampler,
            collate_fn=vqav2.collater,
        )
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map=gpu, torch_dtype=torch.bfloat16)

        inferencer = InferencePipeline(model, gpu, processor)
        scorer = ScoringPipeline()

        processor_kwargs={"padding": True, }
        generate_kwargs={"num_beams": 5, "max_length": 10, "min_length": 1, "length_penalty": -1, "do_sample": False}
        print(generate_kwargs)

        results = inferencer.run_inference(
            dataloader, 
            task="visual_question_answering",
            processor_kwargs=processor_kwargs,
            generate_kwargs=generate_kwargs
        )

        with open(os.path.join(args.output_dir, "%d_results.json" % rank), 'w') as f:
            json.dump(results["answers"], f)
        dist.barrier()

        if rank == 0:
            results = {
                "answers": [],
                "annotations": vqav2.annotation_dict,
                "questions": vqav2.question_dict
            }

            question_ids = set()
            for rank_id in range(world_size):
                with open(os.path.join(args.output_dir, "%d_results.json" % rank_id), 'r') as f:
                    rank_results = json.load(f)
                    for answer in rank_results:
                        question_id = answer["question_id"] 
                        if question_id not in question_ids:
                            results["answers"].append(answer)
                            question_ids.add(question_id)
    
            compute_vqa_results(results, scorer, os.path.join(args.output_dir, "results.json"))
    
        dist.destroy_process_group()
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", load_in_8bit=True, torch_dtype=torch.float16
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inferencer = InferencePipeline(model, device, processor)
        scorer = ScoringPipeline()

        results = inferencer.run_inference(
            vqav2, task="visual_question_answering"
        )

        compute_vqa_results(results, scorer, os.path.join(args.output_dir, "results.json"))
