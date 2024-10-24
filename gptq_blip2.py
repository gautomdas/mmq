import math
import time
from typing import List, Dict, Any, Optional
import argparse
import random
import os
import json

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dataset import COCODataset
import transformers

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ====================================================
# Quantization Classes and Functions
# ====================================================


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


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


def find_linear_layers_in_model(model):
    layers = {}

    def recurse(module, prefix=""):
        if isinstance(module, nn.Linear):
            layers[prefix.rstrip(".")] = module
        for name, child in module.named_children():
            recurse(child, prefix + name + ".")

    recurse(model)
    return layers


# Keep existing GPTQ and Quantizer classes as is...


class BLIP2Quantizer:
    """Main class for quantizing BLIP2 model components with different precisions"""

    def __init__(self, model, processor, device, chunk_size=32):
        self.model = model
        self.processor = processor
        self.device = device
        self.chunk_size = chunk_size

        # Component-specific configuration parameters
        self.config = {
            "vision": {
                "bits": 8,
                "percent_dampening": 0.01,
                "group_size": -1,
                "use_symmetric": True,
                "use_act_order": False,
                "use_static_groups": False,
            },
            "qformer": {
                "bits": 6,
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
            f"\nProcessing {desc} layers {start_idx} to {end_idx-1} with {self.config[component_type]['bits']}-bit precision"
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

    def quantize_vision_model(self, calibration_images):
        """Quantize vision model with 8-bit precision"""
        print(
            f"Quantizing Vision Model with {self.config['vision']['bits']}-bit precision..."
        )
        self.model.vision_model.to(self.device)

        layers = find_linear_layers_in_model(self.model.vision_model)
        total_layers = len(layers)

        def forward_pass():
            for image in tqdm(calibration_images, desc="Processing vision model batch"):
                inputs = self.processor(images=image, return_tensors="pt").to(
                    self.device
                )
                self.model.vision_model(pixel_values=inputs["pixel_values"])

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "vision model", "vision"
            )

        self.model.vision_model.cpu()
        print("Vision Model quantization complete.\n")

    def quantize_qformer(self, calibration_images):
        """Quantize Q-Former with 6-bit precision"""
        print(
            f"Quantizing Q-Former with {self.config['qformer']['bits']}-bit precision..."
        )
        self.model.qformer.to(self.device)
        self.model.vision_model.to(self.device)
        query_tokens = self.model.query_tokens.to(self.device)

        layers = find_linear_layers_in_model(self.model.qformer)
        total_layers = len(layers)

        def forward_pass():
            for image in tqdm(calibration_images, desc="Processing Q-Former batch"):
                inputs = self.processor(images=image, return_tensors="pt").to(
                    self.device
                )
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                image_embeds = vision_outputs.last_hidden_state
                batch_size = image_embeds.shape[0]
                expanded_query_tokens = query_tokens.expand(batch_size, -1, -1)
                self.model.qformer(
                    query_embeds=expanded_query_tokens,
                    encoder_hidden_states=image_embeds,
                )

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "Q-Former", "qformer"
            )

        self.model.qformer.cpu()
        self.model.vision_model.cpu()
        self.model.query_tokens = self.model.query_tokens.cpu()
        print("Q-Former quantization complete.\n")

    def quantize_language_model(self, calibration_images):
        """Quantize language model with 4-bit precision"""
        print(
            f"Quantizing Language Model with {self.config['language']['bits']}-bit precision..."
        )
        self.model.to(self.device)

        layers = find_linear_layers_in_model(self.model.language_model)
        layers["language_projection"] = self.model.language_projection
        total_layers = len(layers)

        def forward_pass():
            for image in tqdm(
                calibration_images, desc="Processing language model batch"
            ):
                inputs = self.processor(images=image, return_tensors="pt").to(
                    self.device
                )
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                image_embeds = vision_outputs.last_hidden_state
                batch_size = image_embeds.shape[0]
                expanded_query_tokens = self.model.query_tokens.expand(
                    batch_size, -1, -1
                )
                qformer_outputs = self.model.qformer(
                    query_embeds=expanded_query_tokens,
                    encoder_hidden_states=image_embeds,
                )
                language_model_inputs = self.model.language_projection(
                    qformer_outputs.last_hidden_state
                )
                attention_mask = torch.ones(
                    language_model_inputs.size()[:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                self.model.language_model(
                    inputs_embeds=language_model_inputs, attention_mask=attention_mask
                )

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "language model", "language"
            )

        self.model.cpu()
        print("Language Model quantization complete.\n")

    def quantize(self, calibration_images):
        """Quantize all BLIP2 components"""
        print("Starting BLIP2 model quantization...")
        self.quantize_vision_model(calibration_images)
        self.quantize_qformer(calibration_images)
        self.quantize_language_model(calibration_images)
        print("BLIP2 model quantization complete.")

    def prepare_for_inference(
        model: Blip2ForConditionalGeneration, device: torch.device
    ) -> None:
        """
        Prepare the BLIP2 model for inference by ensuring consistent device and dtype across components.
        """
        print(f"Preparing model for inference on {device}...")
        dtype = torch.float16  # BLIP2 uses float16 by default

        # Vision Model
        model.vision_model.to(device, dtype)
        # Ensure Conv2d and other layers are properly moved
        for module in model.vision_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data = module.weight.data.to(device, dtype)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(device, dtype)

        # Q-Former
        model.qformer.to(device, dtype)
        model.query_tokens = nn.Parameter(model.query_tokens.to(device, dtype))

        # Language Model and Projection
        model.language_model.to(device, dtype)
        model.language_projection.to(device, dtype)

        print("Model prepared for inference.")


def get_parser():
    parser = argparse.ArgumentParser(description="BLIP2 Quantization Script")

    # Add arguments for bit sizes
    parser.add_argument(
        "--vision-bits",
        type=int,
        default=8,
        choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help="Bit size for vision component",
    )

    parser.add_argument(
        "--qformer-bits",
        type=int,
        default=6,
        choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help="Bit size for qformer component",
    )

    parser.add_argument(
        "--language-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help="Bit size for language component",
    )

    # Add optional arguments for dataset configuration
    parser.add_argument(
        "--calibration-size", type=int, default=128, help="Size of calibration dataset"
    )

    parser.add_argument(
        "--test-size", type=int, default=1024, help="Size of test dataset"
    )

    parser.add_argument(
        "--chunk-size", type=int, default=32, help="Chunk size for processing"
    )

    parser.add_argument(
        "--test-batch-size", type=int, default=128, help="Chunk size for processing"
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
        "--output-dir",
        type=str,
        default="gptq_results",
        help="Directory to save results",
    )

    return parser


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

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

    # Load model and processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Modified model loading without device_map and accelerate
    print("Loading BLIP2 model...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
    )
    # Move model to CPU initially
    model = model.cpu()

    # Free up memory
    torch.cuda.empty_cache()

    # Prepare COCO dataset
    coco_dataset = COCODataset(
        ann_file="./data/coco/annotations/captions_val2017.json",
        img_dir="./data/coco/val2017",
    )

    # Generate random indices for calibration and test sets
    total_indices = list(range(5000))  # Total dataset size

    # Get random calibration indices
    calibration_indices = random.sample(total_indices, args.calibration_size)

    # Remove calibration indices from available indices
    remaining_indices = list(set(total_indices) - set(calibration_indices))

    # Get random test indices
    test_indices = random.sample(remaining_indices, args.test_size)

    # Prepare calibration data
    images = [coco_dataset[i][0] for i in calibration_indices]
    captions = [coco_dataset[i][1] for i in calibration_indices]
    img_ids = [coco_dataset.ids[i] for i in calibration_indices]

    # Create quantizer
    quantizer = BLIP2Quantizer(model, processor, device, chunk_size=args.chunk_size)

    # Update quantizer config with specified bit sizes
    quantizer.config["vision"]["bits"] = args.vision_bits
    quantizer.config["qformer"]["bits"] = args.qformer_bits
    quantizer.config["language"]["bits"] = args.language_bits

    # Print configuration
    print("\nQuantization Configuration:")
    print(f"Vision bits: {args.vision_bits}")
    print(f"QFormer bits: {args.qformer_bits}")
    print(f"Language bits: {args.language_bits}")
    print(f"Calibration size: {args.calibration_size}")
    print(f"Test size: {args.test_size}")
    print(f"Device: {device}\n")

    # Quantize model
    quantizer.quantize(images)

    # Prepare model for inference
    BLIP2Quantizer.prepare_for_inference(model, device)

    # Prepare test data
    test_images = [coco_dataset[i][0] for i in test_indices]
    test_captions = [coco_dataset[i][1] for i in test_indices]
    test_img_ids = [coco_dataset.ids[i] for i in test_indices]

    # Generate captions with quantized model
    print("Generating captions...")
    print(f"Testing images: {len(test_images)}")

    # Process images in chunks to avoid memory issues
    predicted_captions = []

    for i in range(0, len(test_images), args.test_batch_size):
        chunk_images = test_images[i : i + args.chunk_size]
        inputs = processor(images=chunk_images, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                out = model.generate(**inputs)

            chunk_captions = [
                processor.decode(out[j], skip_special_tokens=True).strip()
                for j in range(len(chunk_images))
            ]
            predicted_captions.extend(chunk_captions)

        except RuntimeError as e:
            print(f"Error processing batch {i}: {e}")
            # Move model to CPU and clear cache if we run into memory issues
            model.cpu()
            torch.cuda.empty_cache()
            continue

    # Prepare results
    results = {
        "predictions": [
            {"image_id": img_id, "caption": caption}
            for img_id, caption in zip(test_img_ids, predicted_captions)
        ],
        "references": test_captions,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    filename = (
        f"gptq_blip2_{args.vision_bits}_{args.qformer_bits}_{args.language_bits}.json"
    )
    output_path = os.path.join(args.output_dir, filename)

    print(f"\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved successfully")


if __name__ == "__main__":
    main()

