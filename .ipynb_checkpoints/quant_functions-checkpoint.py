import torch
from torch import Tensor

# curried function for quantization
# uniform_quantization -> num_bits -> Tensor -> result
def uniform_quantization(num_bits):
    def quant(x: Tensor):
        min_val = x.min()
        max_val = x.max()
        
        alpha = max_val - min_val
        x = (x - min_val) / alpha
        
        scale = (2**num_bits - 1)
        result = (scale * x).round()
        result /= scale
        
        result = alpha * result + min_val
        
        return result
    return quant