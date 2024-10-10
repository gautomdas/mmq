import torch
import torch.nn as nn
import gc
import inspect
import functools

def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def compute_loss(fp_output, q_output):
    fp_output_flat = fp_output.view(-1)
    q_output_flat = q_output.view(-1)
    L2 = torch.linalg.norm(fp_output_flat - q_output_flat, ord=2)
    return L2

# returns all nn.linear within module (a layer)
def get_named_linears(module):
    return {name: mod for name, mod in module.named_modules() if isinstance(mod, nn.Linear)}

def get_mods(model, non_linears_only = True):
    children = list(model.children())

    if non_linears_only:
        return [model] if len(children) == 0 else [ci for c in children for ci in get_mods(c, non_linears_only) if type(ci) != torch.nn.Linear]
    else:
        return [model] if len(children) == 0 else [ci for c in children for ci in get_mods(c, non_linears_only)]


def sanitize_kwargs(inputs_kwargs, module):
    """
    Remove the arguments that are not supported in the module's
    forward pass

    Args:
        inputs_kwargs (`dict`):
            The input dictionary to pass to the model layer
        module (`torch.nn.Module`):
            Target module to quantize.
    """
    module_signature = inspect.signature(module.forward).parameters
    sanitized_kwargs = {}
    for k, v in inputs_kwargs.items():
        if k in module_signature:
            sanitized_kwargs[k] = v
    return sanitized_kwargs


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))