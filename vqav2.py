import os
import json

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from transformers import Blip2ForConditionalGeneration, Blip2Processor 

from datasets import VQAv2Eval
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline

distributed = True
num_workers = 1
batch_size = 64
result_dir = "./vqa_results"

def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu)

    return rank, world_size, gpu

def compute_vqa_results(results, scorer, save_path=None):
    vqa_results = scorer.compute_scores(results, "visual_question_answering")
    print(vqa_results)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(vqa_results, f)

if __name__ == "__main__":
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    vqav2 = VQAv2Eval(
        "./data/vqav2/val2014",
        "./data/vqav2/annotations",
        "./data/vqav2/questions",
    )

    if distributed:
        rank, world_size, gpu = init_distributed()
        dist.barrier()

        sampler = DistributedSampler(
            vqav2,
            shuffle=False,
            num_replicas=world_size,
            rank=rank
        )
        # Create DataLoader
        dataloader = DataLoader(
            vqav2, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=False,
            shuffle=False,
            sampler=sampler,
            collate_fn=vqav2.collater,
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", load_in_8bit=True, torch_dtype=torch.float16, device_map=gpu
        )

        inferencer = InferencePipeline(model, gpu, processor)
        scorer = ScoringPipeline()

        results = inferencer.run_inference(
            dataloader, task="visual_question_answering"
        )

        with open(os.path.join(result_dir, "%d_results.json" % rank), 'w') as f:
            json.dump({"answers": results["answers"]}, f)
        dist.barrier()

        if rank == 0:
            results = {
                "answers": [],
                "annotations": vqav2.annotation_dict,
                "questions": vqav2.question_dict
            }

            for rank_id in range(world_size):
                with open(os.path.join(result_dir, "%d_results.json" % rank_id), 'r') as f:
                    rank_results = json.load(f)
                    results["answers"] += rank_results["answers"]
    
            compute_vqa_results(results, scorer, os.path.join(result_dir, "results.json"))
    
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

        compute_vqa_results(results, scorer, os.path.join(result_dir, "results.json"))
