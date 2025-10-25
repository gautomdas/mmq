from enum import Enum, auto
from typing import Callable, List

import torch
from torch import nn


class ModelPart(Enum):
    VIT = auto()
    QFORMER = auto()
    LLM = auto()

class LayerGroup(Enum):
    FIRST = auto()
    MIDDLE = auto()
    LAST = auto()
    ALL = auto()

class LayerType(Enum):
    MLP = auto()
    ATTENTION = auto()
    BOTH = auto()


class QuantConfig:
    def __init__(
        self,
        model_part: ModelPart,
        layer_group: LayerGroup,
        layer_type: LayerType,
        quant_function: Callable,
        num_bits: int,
    ):
        self.model_part = model_part
        self.layer_group = layer_group
        self.layer_type = layer_type
        self.quant_function = quant_function
        self.num_bits = num_bits

class BlipQuantizer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.num_bits = 0

    def apply_quantization(self, configs: List[QuantConfig]):
        for config in configs:
            self._quantize_part(config)

    def _quantize_part(self, config: QuantConfig):
        if config.model_part == ModelPart.VIT:
            layers = self.model.vision_model.encoder.layers
        elif config.model_part == ModelPart.QFORMER:
            layers = self.model.qformer.encoder.layer
        else:  # LLM
            layers = self.model.language_model.model.decoder.layers

        total_layers = len(layers)
        start, end = self._get_layer_range(config.layer_group, total_layers)

        self.num_bits = config.num_bits
        # self.print(f"running {self.num_bits} quant")
        bit_quant_function = config.quant_function(config.num_bits)

        for layer in layers[start:end]:
            if config.layer_type in [LayerType.MLP, LayerType.BOTH]:
                self._quantize_mlp(layer, bit_quant_function)
            if config.layer_type in [LayerType.ATTENTION, LayerType.BOTH]:
                self._quantize_attention(layer, bit_quant_function)

    def _get_layer_range(self, group: LayerGroup, total_layers: int):
        if group == LayerGroup.FIRST:
            return 0, total_layers // 3
        elif group == LayerGroup.LAST:
            return 2 * total_layers // 3, total_layers
        elif group == LayerGroup.MIDDLE:
            return total_layers // 3, 2 * total_layers // 3
        else:  # ALL
            return 0, total_layers

    def _quantize_mlp(self, layer: nn.Module, quant_function: Callable):
        if hasattr(layer, "mlp"):
            self._quantize_linear(layer.mlp.fc1, quant_function)
            self._quantize_linear(layer.mlp.fc2, quant_function)
        elif hasattr(layer, "fc1") and hasattr(layer, "fc2"):
            self._quantize_linear(layer.fc1, quant_function)
            self._quantize_linear(layer.fc2, quant_function)

    def _quantize_attention(self, layer: nn.Module, quant_function: Callable):
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "qkv"):
                self._quantize_linear(layer.self_attn.qkv, quant_function)
            if hasattr(layer.self_attn, "projection"):
                self._quantize_linear(layer.self_attn.projection, quant_function)
        elif hasattr(layer, "attention"):
            if hasattr(layer.attention, "attention"):
                self._quantize_linear(layer.attention.attention.query, quant_function)
                self._quantize_linear(layer.attention.attention.key, quant_function)
                self._quantize_linear(layer.attention.attention.value, quant_function)
            if hasattr(layer.attention, "output"):
                self._quantize_linear(layer.attention.output.dense, quant_function)
        elif hasattr(layer, "k_proj"):
            self._quantize_linear(layer.k_proj, quant_function)
            self._quantize_linear(layer.v_proj, quant_function)
            self._quantize_linear(layer.q_proj, quant_function)
            self._quantize_linear(layer.out_proj, quant_function)

    def _quantize_linear(self, module: nn.Module, quant_function: Callable):
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            module.weight.data = quant_function(module.weight.data)
            module.quantized = True
            module.num_bits = self.num_bits
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            module.bias.data = quant_function(module.bias.data)

    def count_quantized_layers(self):
        count = 0
        for name, module in self.model.named_modules():
            if hasattr(module, "quantized") and module.quantized:
                count += 1
        return count

    def get_bits(self):
        return self.num_bits

    def print_model_structure(self, indent=0):
        for name, module in self.model.named_children():
            print("  " * indent + name + ": " + module.__class__.__name__, end="")
            if hasattr(module, "quantized"):
                print(f" (Quantized: {module.num_bits} bits)", end="")
            print()
            if len(list(module.children())) > 0:
                self.print_model_structure(indent + 1)
