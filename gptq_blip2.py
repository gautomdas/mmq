import math
import time
from typing import Union

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformers import (Blip2ForConditionalGeneration,
                          Blip2ForImageTextRetrieval)

from utils import get_calibration_set

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

    def __init__(self, model, processor, device, config, chunk_size=32):
        self.model = model
        self.processor = processor
        self.device = device
        self.chunk_size = chunk_size
        self.config = config

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
        self.model.vision_model.to(self.device)

        layers = find_linear_layers_in_model(self.model.vision_model)
        total_layers = len(layers)

        def forward_pass():
            for i in tqdm(
                range(len(calibration_set)),
                # range(len(calibration_set["images"])),
                desc="Processing vision model batch",
            ):
                inputs = self.processor(
                    **calibration_set[i],
                    # images=calibration_set["images"][i],
                    # text=calibration_set["text_input"][i],
                    return_tensors="pt",
                ).to(self.device)
                self.model.vision_model(pixel_values=inputs["pixel_values"])

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "vision model", "vision"
            )

        self.model.vision_model.cpu()
        print("Vision Model quantization complete.\n")

    def quantize_qformer(self, calibration_set, task):
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
            for i in tqdm(
                range(len(calibration_set)),
                desc="Processing Q-Former batch",
                # range(len(calibration_set["images"])), desc="Processing Q-Former batch"
            ):
                inputs = self.processor(
                    **calibration_set[i],
                    # images=calibration_set["images"][i],
                    # text=calibration_set["text_input"][i],
                    return_tensors="pt",
                ).to(self.device)

                if task == "image_text_retrieval":
                    self.model(**inputs, use_image_text_matching_head=True)
                    return

                image_embeds = self.model.vision_model(
                    pixel_values=inputs["pixel_values"],
                    return_dict=True,
                    interpolate_pos_encoding=False,
                ).last_hidden_state
                image_embeds = image_embeds.to(self.device)
                image_attention_mask = torch.ones(
                    image_embeds.size()[:-1],
                    dtype=torch.long,
                    device=image_embeds.device,
                ).to(self.device)

                query_tokens = self.model.query_tokens.expand(
                    image_embeds.shape[0], -1, -1
                ).to(self.device)
                self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                """
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
                """

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "Q-Former", "qformer"
            )

        self.model.qformer.cpu()
        self.model.vision_model.cpu()
        self.model.query_tokens = self.model.query_tokens.cpu()
        print("Q-Former quantization complete.\n")

    def quantize_language_model(self, calibration_set):
        """Quantize language model with 4-bit precision"""
        print(
            f"Quantizing Language Model with {self.config['language']['bits']}-bit precision..."
        )
        self.model.to(self.device)

        layers = find_linear_layers_in_model(self.model.language_model)
        layers["language_projection"] = self.model.language_projection
        total_layers = len(layers)

        def forward_pass():
            for i in tqdm(
                range(len(calibration_set)),
                # range(len(calibration_set["images"])),
                desc="Processing language model batch",
            ):
                inputs = self.processor(
                    **calibration_set[i],
                    return_tensors="pt",
                ).to(self.device)

                self.model.generate(**inputs)
                """
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
                """

        for start_idx in range(0, total_layers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_layers)
            self._process_chunk(
                layers, start_idx, end_idx, forward_pass, "language model", "language"
            )

        self.model.cpu()
        print("Language Model quantization complete.\n")

    def quantize(self, dataset, task):
        """Quantize all BLIP2 components"""
        print("Starting BLIP2 model quantization...")
        calibration_set = get_calibration_set(dataset)

        self.quantize_vision_model(calibration_set)
        self.quantize_qformer(calibration_set, task)
        if task != "image_text_retrieval":
            self.quantize_language_model(calibration_set)

        print("BLIP2 model quantization complete.")

    @staticmethod
    def prepare_for_inference(
        self,
        model: Union[Blip2ForConditionalGeneration, Blip2ForImageTextRetrieval],
        device: torch.device,
    ) -> None:
        """
        Prepare the BLIP2 model for inference by ensuring consistent device and dtype across components.
        """
        print(f"Preparing model for inference on {device}...")
        dtype = torch.float16  # BLIP2 uses float16 by default

        # Vision Model
        self.model.vision_model.to(self.device, dtype)
        # Ensure Conv2d and other layers are properly moved
        for module in self.model.vision_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data = module.weight.data.to(self.device, dtype)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(self.device, dtype)

        # Q-Former
        self.model.qformer.to(self.device, dtype)
        self.model.query_tokens = nn.Parameter(self.model.query_tokens.to(self.device, dtype))

        # Language Model and Projection
        if isinstance(self.model, Blip2ForConditionalGeneration):
            self.model.language_model.to(self.device, dtype)
            self.model.language_projection.to(self.device, dtype)

        print("Model prepared for inference.")
