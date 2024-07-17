from torch import nn
import json
from blip_quantizer import QuantConfig, ModelPart, LayerGroup, LayerType
from quant_functions import uniform_quantization

def load_quant_configs(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    quant_configs = []
    for q in config:
        quant_configs.append(QuantConfig(
            model_part=ModelPart[q['model_part']],
            layer_group=LayerGroup[q['layer_group']],
            layer_type=LayerType[q['layer_type']],
            quant_function=uniform_quantization,
            num_bits=q['num_bits']
        ))
    
    return quant_configs
    
def save_quant_configs(configs, filename):
    json_configs = []
    for config in configs:
        json_config = {
            "model_part": config.model_part.name,
            "layer_group": config.layer_group.name,
            "layer_type": config.layer_type.name,
            "num_bits": config.num_bits
        }
        json_configs.append(json_config)
    
    with open(filename, 'w') as f:
        json.dump(json_configs, f, indent=2)

def print_model_structure(model:nn.Module, indent=0):
    for name, module in model.named_children():
        print('  ' * indent + name + ': ' + module.__class__.__name__, end='')
        if hasattr(module, 'quantized'):
            print(f" (Quantized: {module.num_bits} bits)", end='')
        print()
        if len(list(module.children())) > 0:
            print_model_structure(module, indent + 1)