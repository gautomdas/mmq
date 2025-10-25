import builtins as __builtin__
import json
import os
import random

import torch
import torch.distributed as dist
from torch import nn

from blip_quantizer import LayerGroup, LayerType, ModelPart, QuantConfig
from dataset import COCODataset, Flickr30kEvalDataset, GQAEval, VQAv2Eval
from quant_functions import uniform_quantization


def load_quant_configs(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    quant_configs = []
    for q in config:
        quant_configs.append(
            QuantConfig(
                model_part=ModelPart[q["model_part"]],
                layer_group=LayerGroup[q["layer_group"]],
                layer_type=LayerType[q["layer_type"]],
                quant_function=uniform_quantization,
                num_bits=q["num_bits"],
            )
        )

    return quant_configs


def save_quant_configs(configs, filename):
    json_configs = []
    for config in configs:
        json_config = {
            "model_part": config.model_part.name,
            "layer_group": config.layer_group.name,
            "layer_type": config.layer_type.name,
            "num_bits": config.num_bits,
        }
        json_configs.append(json_config)

    with open(filename, "w") as f:
        json.dump(json_configs, f, indent=2)


def print_model_structure(model: nn.Module, indent=0):
    for name, module in model.named_children():
        print("  " * indent + name + ": " + module.__class__.__name__, end="")
        if hasattr(module, "quantized"):
            print(f" (Quantized: {module.num_bits} bits)", end="")
        print()
        if len(list(module.children())) > 0:
            print_model_structure(module, indent + 1)


def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ("WORLD_SIZE"))
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


def compute_results(results, scorer, task, save_path=None):
    results = scorer.compute_scores(results, task)
    print(results)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f)


def gather_results(args, rank, world_size):
    if args.task == "image_captioning":
        results = {"predictions": [], "references": []}
    elif args.task == "vqav2":
        results = {
            "answers": [],
            "annotations": os.path.join(
                args.dataset_dir,
                "annotations/v2_mscoco_val2014_annotations.json",
            ),
            "questions": os.path.join(
                args.dataset_dir,
                "questions/v2_OpenEnded_mscoco_val2014_questions.json",
            ),
        }
    elif args.task == "gqa":
        results = []

    id_set = set()
    for rank_id in range(world_size):
        with open(os.path.join(args.output_dir, f"{rank_id}_results.json", "r")) as f:
            rank_results = json.load(f)

            if args.task == "image_captioning":
                for prediction, reference in zip(
                    rank_results["predictions"], rank_results["references"]
                ):
                    image_id = prediction["image_id"]
                    if image_id not in id_set:
                        results["predictions"].append(prediction)
                        results["references"].append(reference)
                        id_set.add(image_id)
            elif args.task == "vqav2":
                for answer in rank_results:
                    question_id = answer["question_id"]
                    if question_id not in id_set:
                        results["answers"].append(answer)
                        id_set.add(question_id)
            elif args.task == "gqa":
                for answer in rank_results:
                    question_id = answer["question_id"]
                    if question_id not in id_set:
                        results.append(answer)
                        id_set.add(question_id)

    return results


def get_calibration_set(dataset, calibration_size=128):
    # calibration_set = {}
    calibration_set = []
    calibration_indices = random.sample(range(len(dataset)), calibration_size)
    calibration_data = [dataset[i] for i in calibration_indices]

    if isinstance(dataset, COCODataset):
        # calibration_set["images"] = [sample["image"] for sample in calibration_data]
        calibration_set = [
            {"images": sample["image"]} for sample in calibration_data
        ]
    elif isinstance(dataset, Flickr30kEvalDataset):
        # calibration_set["images"] = [sample["image"] for sample in calibration_data]
        # calibration_set["text_input"] = [
        #     dataset.text[sample["index"]] for sample in calibration_data
        # ]
        calibration_set = [
            {"images": sample["image"], "text_input": dataset.text[sample["index"]]}
            for sample in calibration_data
        ]
    elif isinstance(dataset, GQAEval) or isinstance(dataset, VQAv2Eval):
        # calibration_set["images"] = [sample["image"] for sample in calibration_data]
        # calibration_set["text_input"] = [
        #     sample["text_input"] for sample in calibration_data
        # ]
        calibration_set = [
            {"images": sample["image"], "text_input": sample["text_input"]}
            for sample in calibration_data
        ]
    else:
        return ValueError("Invalid dataset type")

    return calibration_set
