import sys

sys.path.append("..")
import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.models.llava.image_processing_llava import \
    LlavaImageProcessor

from dataset import GQAEval, VQAv2Eval
from inference_pipeline import InferencePipeline
from scoring_pipeline import ScoringPipeline

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

#

# ====================================================
# Quantization Classes and Functions
# ====================================================


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def find_linear_layers_in_model(model):
    layers = {}

    def recurse(module, prefix=""):
        if isinstance(module, nn.Linear):
            layers[prefix.rstrip(".")] = module
        for name, child in module.named_children():
            recurse(child, prefix + name + ".")

    recurse(model)
    return layers


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        device = self.maxq.device
        self.maxq = torch.tensor(2**bits - 1, device=device)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1, device=device)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

        # Ensure buffers are on the same device as input x
        self.scale = self.scale.to(dev)
        self.zero = self.zero.to(dev)

    def quantize(self, x):
        if self.ready():
            # Ensure buffers are on the same device as x
            self.scale = self.scale.to(x.device)
            self.zero = self.zero.to(x.device)
            self.maxq = self.maxq.to(x.device)
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.quantizer.to(self.dev)

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)], weight=True
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("Time for quantization: %.2f seconds" % (time.time() - tick))
        print("Total quantization error:", torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


class LlavaQuantizer:
    def __init__(self, model, processor, device, config, chunk_size=32, task="vqav2"):
        self.model = model
        self.processor = processor
        self.device = device
        self.chunk_size = chunk_size
        self.task = task
        self.config = config

        # Component-specific configuration parameters
        self.config = {
            "vision": {
                "bits": 4,
                "percent_dampening": 0.01,
                "group_size": -1,
                "use_symmetric": True,
                "use_act_order": False,
                "use_static_groups": False,
            },
            "language": {
                "bits": 4,
                "percent_dampening": 0.01,
                "group_size": -1,
                "use_symmetric": True,
                "use_act_order": False,
                "use_static_groups": False,
            },
        }

    def _prepare_quantizers(self, layers, component_type):
        """Initialize GPTQ quantizers for given layers with component-specific settings"""
        config = self.config[component_type]
        quantizers = {}
        for name, layer in layers.items():
            quantizers[name] = GPTQ(layer)
            quantizers[name].quantizer.configure(
                bits=config["bits"],
                perchannel=True,
                sym=config["use_symmetric"],
                mse=False,
            )
        return quantizers

    def _process_chunk(
        self, layers, start_idx, end_idx, forward_func, desc, component_type
    ):
        """Process a chunk of layers with component-specific quantization settings"""
        current_layers = dict(list(layers.items())[start_idx:end_idx])
        print(
            f"\nProcessing {desc} layers {start_idx} to {end_idx - 1} with {self.config[component_type]['bits']}-bit precision"
        )

        # Initialize quantizers for current chunk
        quantizers = self._prepare_quantizers(current_layers, component_type)
        hooks = []

        def get_hook(name):
            def hook(module, inp, out):
                if name in quantizers:
                    quantizers[name].add_batch(inp[0].detach(), out.detach())

            return hook

        for name, layer in current_layers.items():
            hooks.append(layer.register_forward_hook(get_hook(name)))

        forward_func()

        for hook in hooks:
            hook.remove()

        config = self.config[component_type]
        for name, layer in current_layers.items():
            print(f"Quantizing layer {name}...")
            quantizer = quantizers[name]
            quantizer.fasterquant(
                blocksize=32,
                percdamp=config["percent_dampening"],
                groupsize=config["group_size"],
                actorder=config["use_act_order"],
                static_groups=config["use_static_groups"],
            )

            layer.weight.data = quantizer.quantizer.quantize(layer.weight.data).to(
                layer.weight.data.dtype
            )
            quantizer.free()

        torch.cuda.empty_cache()

    def quantize_vision_model(self, calibration_set):
        """Quantize vision model with 8-bit precision"""
        print(
            f"Quantizing Vision Model with {self.config['vision']['bits']}-bit precision..."
        )

        # some extra components need to be on device for vision model forward pass
        # self.model.vision_tower.to(self.device)
        self.model.to(self.device)
        self.model.language_model.to("cpu")

        layers = find_linear_layers_in_model(self.model.vision_tower.vision_model)
        total_layers = len(layers)

        print(f"total_layers: {total_layers}")
        print(layers)

        def forward_pass():
            vision_feature_layer = self.model.config.vision_feature_layer
            vision_feature_select_strategy = (
                self.model.config.vision_feature_select_strategy
            )
            image_sizes = None

            # TODO: adjust for GQA if needed
            if self.task == "vqav2":
                for img, prompt in tqdm(
                    calibration_set, desc="Processing vision model batch"
                ):
                    inputs = self.processor(
                        images=[img], text=[prompt], return_tensors="pt", padding=True
                    ).to(self.device)

                    # runs forward pass through vision_tower
                    self.model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        vision_feature_layer=vision_feature_layer,
                        vision_feature_select_strategy=vision_feature_select_strategy,
                        image_sizes=image_sizes,
                    )

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "vision model", "vision"
            )

        self.model.vision_tower.vision_model.cpu()
        print("Vision Model quantization complete.\n")

    def quantize_language_model(self, calibration_set):
        """Quantize language model with 4-bit precision"""
        print(
            f"Quantizing Language Model with {self.config['language']['bits']}-bit precision..."
        )
        self.model.to(self.device)

        layers = find_linear_layers_in_model(self.model.language_model.model)
        # layers["language_projection"] = self.model.language_projection
        total_layers = len(layers)

        def forward_pass():
            # TODO: adjust for GQA if needed
            if self.task == "vqav2":
                for img, prompt in tqdm(
                    calibration_set, desc="Processing language model batch"
                ):
                    inputs = self.processor(
                        images=[img], text=[prompt], return_tensors="pt", padding=True
                    ).to(self.device)

                    self.model.generate(**inputs)

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "language model", "language"
            )

        self.model.cpu()
        print("Language Model quantization complete.\n")

    def quantize(self, calibration_set):
        """Quantize all LLAVA components"""
        print("Starting LLAVA model quantization...")
        self.quantize_vision_model(calibration_set)
        self.quantize_language_model(calibration_set)
        print("LLAVA model quantization complete.")

    def prepare_for_inference(self):
        self.model.to(self.device)


def get_args():
    parser = argparse.ArgumentParser(description="LLAVA GPTQ Quantization Script")

    parser.add_argument(
        "--task",
        type=str,
        choices=["vqav2", "gqa"],
        required=True,
        help="task to evaluate GPTQ-quantized LLAVA on",
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
        default="gptq_results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--no_quant",
        default=False,
        action="store_true",
        help="Set to true to apply no quantization (full-precision run)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size for task evaulation"
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
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16
    )
    # offload model to cpu for now
    model.to("cpu")
    # Free up memory
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", pad_token="<pad>", use_fast=False
    )

    # need to use this image processor w/ do_pad=True according to "Note regarding reproducing original implementation"
    # https://huggingface.co/docs/transformers/en/model_doc/llava
    image_processor = LlavaImageProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", do_pad=True
    )

    processor.image_processor = image_processor

    # short answer prompting according to: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
    llava_prompt = "USER: <image>\n{}\nAnswer the question using a single word or phrase. ASSISTANT:"

    if args.task == "vqav2":
        # VQAv2 dataset paths
        ann_root = "./data/vqav2/annotations"
        q_root = "./data/vqav2/questions"
        image_root = "./data/vqav2/val2014"

        dataset = VQAv2Eval(
            image_root=image_root, ann_root=ann_root, q_root=q_root, prompt=llava_prompt
        )

        dataset.set_max_samples(21435)

    elif args.task == "gqa":
        # GQA dataset paths
        image_root = "./data/gqa/images"
        q_root = "./data/gqa/questions"

        dataset = GQAEval(image_root, q_root, prompt=llava_prompt)

    # Get random calibration indices
    total_indices = list(range(len(dataset)))  # Total dataset size
    calibration_indices = random.sample(total_indices, args.calibration_size)
    calibration_set = [
        (dataset[i]["image"], dataset[i]["text_input"]) for i in calibration_indices
    ]

    # Create quantizer
    quantizer = LlavaQuantizer(model, processor, device)

    if not args.no_quant:
        # Update quantizer config with specified bit sizes
        quantizer.config["vision"]["bits"] = args.vision_bits
        quantizer.config["language"]["bits"] = args.language_bits

        # Print configuration
        print("\nQuantization Configuration:")
        print(f"Vision bits: {args.vision_bits}")
        print(f"Language bits: {args.language_bits}")
        print(f"Calibration size: {args.calibration_size}")
        print(f"Device: {device}\n")

        # Quantize model
        quantizer.quantize(calibration_set)

    # Evaluate on task
    gpu_name = torch.cuda.get_device_name()
    print(gpu_name)

    # adjust batch sizes depending on available gpu memory
    if "A5000" in gpu_name.replace(" ", ""):
        args.batch_size = 16
    elif "A6000" in gpu_name.replace(" ", ""):
        args.batch_size = 56

    print(f"Evaluating on task: {args.task}")
    print(f"batch_size: {args.batch_size}")
    quantizer.prepare_for_inference()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        collate_fn=dataset.collater,
    )

    inferencer = InferencePipeline(model, device, processor)

    # set this according to huggingface usage tips: https://huggingface.co/docs/transformers/en/model_doc/llava
    processor.tokenizer.padding_side = "left"
    processor_kwargs = dict(padding=True)

    # greedy decoding
    generate_kwargs = {"num_beams": 1, "do_sample": False}

    results = inferencer.run_inference(
        dataloader,
        task=args.task,
        processor_kwargs=processor_kwargs,
        generate_kwargs=generate_kwargs,
    )

    json_out = {
        "answers": results,
        "vision_bits": args.vision_bits,
        "language_bits": args.language_bits,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(
        args.output_dir, f"results_v{args.vision_bits}_l{args.language_bits}.json"
    )
    with open(json_path, "w") as f:
        json.dump(json_out, f)

    print(f"Output results to {json_path}")

    # if args.task == 'vqav2':
    #     results["annotations"] = os.path.join(ann_root, "v2_mscoco_val2014_annotations.json")
    #     results["questions"] = os.path.join(q_root, "v2_OpenEnded_mscoco_val2014_questions.json")

    # scorer = ScoringPipeline()

    # def compute_vqa_results(results, scorer, save_path=None):
    #     vqa_results = scorer.compute_scores(results, "vqav2")
    #     print(vqa_results)
    #     if save_path:
    #         with open(save_path, "w") as f:
    #             json.dump(vqa_results, f)

    # compute_vqa_results(results, scorer, os.path.join(args.output_dir, "results.json"))


if __name__ == "__main__":
    main()
