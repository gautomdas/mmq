import torch
import numpy as np
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2ForImageTextRetrieval

from tqdm import tqdm
from collections import defaultdict
from functools import partial
from typing import Tuple, List
import random

from awq.scaled_modules import ScaledModule
from awq.utils import *

# ====================================================
# Base AWQ Quantizer Class
# ====================================================
class BaseAWQQuantizer():
    
    def __init__(self, model, device, inputs_processor, dataset, **kwargs):
        self.model = model
        self.device = device
        self.inputs_processor = inputs_processor
        self.dataset = dataset

        # QUANTIZATION SETTINGS
        self.w_bits = 4
        self.group_size = 128
        self.grid_search_size = 20
        self.zero_point = True

        # Calibration set size, AutoAWQ uses 128 for LLMs
        self.n_samples = 128
        self.run_model = None

    
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bits- 1) - 1
            min_int = -(2 ** (self.w_bits - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros


    @torch.no_grad
    def quantize(self):
        '''
            Apply AWQ to self.model, pseudo-quantizing weights in-place
        '''

        layer_groups = self._get_model_layer_groups()
        calibration_set = self._get_calibration_set()

        # Run calibration set through model
        first_inputs, self.layer_args, self.layer_kwargs = self._gather_first_inputs(layer_groups, calibration_set)

        for layer_group, modules in layer_groups.items():
            self.inps = first_inputs[layer_group]

            for i in tqdm(range(len(modules)), desc= f"Quantizing {layer_group}"):

                layer = modules[i]
                layer = layer.to(self.device)

                # nn.linear modules within layer to quantize
                named_linears = get_named_linears(layer)
                linear_inputs = self._gather_linear_inputs(layer, named_linears, layer_group)
                grouped_mods = self._group_modules_for_scaling(layer, linear_inputs, layer_group)

                # compute scales over each group of modules to quantize
                # TODO: filter grouped_mods according to our quantization config
                scales = [
                    self._compute_scales(layer, **group)
                    for group in grouped_mods    
                ]

                # apply scales to prev_op and modules
                for group, scale in zip(grouped_mods, scales):                
                    assert torch.all(scale)
                    self._apply_scales(scale, group['prev_op'], group['modules'], layer)


                # solve for and apply clipping
                clips = self._search_best_clip(named_linears, linear_inputs)
                self._apply_clip(named_linears, clips)

                # apply pseudo_quant to linear weights
                for name, module in named_linears.items():
                    # module = module.to(device).half()
                    module = module.to(self.device)
                    module.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                        module.weight.data
                    )


    def _gather_first_inputs(self, layer_groups, calibration_set):
        '''
            Gather initial inputs (+other positional args, kwargs) to each layer group
            Runs calibration set through model up until the last layer group
        '''

        first_inputs = {}
        layer_args = {}
        layer_kwargs = {}

        # get input and kwargs to layer 0 (for each group of layers)
        # use this Catcher hack cause forward hooks cannot capture kwargs
        class Catcher(nn.Module):
            def __init__(self, module, layer_group, is_last):
                super().__init__()
                self.module = module
                self.layer_group = layer_group
                self.is_last = is_last

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    # del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                first_inputs[self.layer_group] = hidden_states

                # preserve rest of positional arguments
                layer_args[self.layer_group] = args[1:]
                layer_kwargs[self.layer_group] = kwargs
                
                # early exit for last group of layers
                if self.is_last:
                    raise ValueError

                return self.module.forward(*args, **kwargs)

        keys = list(layer_groups.keys())

        for i in range(len(keys)):
            layer_group = keys[i]
            is_last = True if i == len(keys) - 1 else False

            modules = layer_groups[layer_group]
            modules[0] = Catcher(modules[0], layer_group, is_last)

        self.model = self.model.to(self.device)
        calibration_set = calibration_set.to(self.device)

        # NOTE: catching raised ValueError to stop inference early
        try:
            self.run_model(calibration_set)
        except ValueError:
            pass
        
        calibration_set = calibration_set.cpu()
        clear_memory(calibration_set)

        for _, modules in layer_groups.items():
            # restore proper module at beginning of layer group
            modules[0] = modules[0].module
        
        return first_inputs, layer_args, layer_kwargs
       

    def _gather_linear_inputs(self, layer, named_linears, layer_group):
        '''
            Gather inputs to linear layers using pytorch forward hooks
        '''

        def input_hook(module, input, output, module_name, inputs):
            x = input[0]
            x = x.detach().cpu()
            inputs[module_name].append(x)
        

        inputs = defaultdict(list)
        hooks = []
        
        for name, mod in named_linears.items():
            hooks.append(
                mod.register_forward_hook(partial(input_hook,
                                                  module_name = name, 
                                                  inputs = inputs))
            )

        # compute next set of inputs, grabbing linear inputs through the hooks
        self.inps = layer(self.inps, *self.layer_args[layer_group], **self.layer_kwargs[layer_group])
        self.inps = self.inps[0]

        # remove hooks from model
        for hook in hooks:
            hook.remove()

        inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}

        return inputs
    
    def _compute_scales(self, layer, prev_op, modules, inp, parent_module, layer_kwargs):

        '''
            Grid search for scales to preserve salient weights,
            Minimizes L2 loss between full-precision and quantized output
        '''
        
        inp = inp.to(self.device)

        # block of weights concatted together
        W = torch.cat([mod.weight for mod in modules], dim = 0)
        orig_shape = W.shape
        W = W.view(-1, self.group_size)

        # rescale W to 0-1 scale
        W_scale = W.abs() / (W.abs().amax(dim=1, keepdim=True) + 1e-6)
        W_scale = W_scale.view(orig_shape)
        # per channel mean of normalized weights
        W_mean = W_scale.mean(0)
        W_mean = W_mean.view(-1)

        clear_memory(W)

        # per channel mean of input (activation)
        X_mean = inp.abs().view(-1, inp.shape[-1]).mean(0)
        X_mean = X_mean.view(-1)

        kwargs = sanitize_kwargs(layer_kwargs, parent_module)

        # compute full precision output
        with torch.no_grad():
            fp_output = parent_module(inp, **kwargs)[0]

        
        # Grid search for best scales
        n_grid = self.grid_search_size
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in parent_module.state_dict().items()}

        for ratio in range(n_grid):
            scales = X_mean.pow(ratio).clamp(min=1e-4).view(-1)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            scales_view = scales.view(1, -1).to(self.device)

            # Q(W * s)
            # pseudo-quantize modules (nn.linear)
            for mod in modules:
                mod.weight.mul_(scales_view)
                mod.weight.data = (
                    self.pseudo_quantize_tensor(mod.weight.data)[0] / scales_view
                )

            with torch.no_grad():
                # Q(W * s) * X
                q_output = parent_module(inp, **kwargs)[0]
            
            # Compute loss (L2 NORM)
            loss = compute_loss(fp_output, q_output)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            # reset to original weights
            parent_module.load_state_dict(org_sd)

        assert best_ratio != -1, "best scales ratio never set"
        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()
    

    def _apply_scales(self, scale, prev_op, modules, layer):

        '''
            Applies scales to weights in modules and fuses scales
            to prev_op/adds wrapper module
        '''

        scale = scale.to(self.device)
        
        # define custom pytorch wrapper module to apply scaling for input
        # doing this when there isn't a convenient module to fuse scaling with prior to reaching modules
        if isinstance(prev_op, str):
            module = rgetattr(layer, prev_op)
            scaled_mod = ScaledModule(scale, module)
            module = rsetattr(layer, prev_op, scaled_mod)

        else:
            prev_op = prev_op.to(self.device)
            
            # fuse scales with previous LayerNorm
            if isinstance(prev_op, torch.nn.LayerNorm):
                prev_op.weight.div_(scale)

                if hasattr(prev_op, "bias") and prev_op.bias is not None:
                    prev_op.bias.div_(scale)

            # fuse scales with previous Linear module
            elif isinstance(prev_op, torch.nn.Linear):
                prev_op.weight[-scale.size(0) :].div_(scale.view(-1, 1))

            # store (W*s)
            for fc in modules:
                fc.weight.mul_(scale.view(1, -1))

            # SANITY checks
            for p in prev_op.parameters():
                assert torch.isnan(p).sum() == 0
            for fc in modules:
                for p in fc.parameters():
                    assert torch.isnan(p).sum() == 0

            prev_op.cpu()
            for fc in modules:
                fc.cpu()
            scale.cpu()


    @torch.no_grad()
    def _apply_clip(self, modules, clip_list: Tuple[str, torch.Tensor]):
        '''
            Clamp outlier values before quantization according to computed clipping values
        '''
        for name, max_val in clip_list:
            module: nn.Linear = modules[name]
            module.to(self.device)

            max_val = max_val.to(module.weight.device)
            org_shape = module.weight.shape

            module.weight.data = module.weight.data.reshape(*max_val.shape[:2], -1)
            module.weight.data = torch.clamp(module.weight.data, -max_val, max_val)
            module.weight.data = module.weight.data.reshape(org_shape)
            module.cpu()

    def _search_best_clip(self, modules, linear_inputs):
        '''
            Grid search for best clipping ranges for each module in modules
        '''

        clip_list = []

        # NOTE: awq libraries seem to avoid clipping attention modules
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in modules:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            modules[name].to(self.device)
            max_val = self._compute_best_clip(
                modules[name].weight, linear_inputs[name]
            )
            clip_list.append((name, max_val))
            modules[name].cpu()

        return clip_list


    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        
        '''
            Compute best clipping ranges (grid search) for w
            Minimizes MSE b/w original and quantized values
        '''

        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size # if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def _get_calibration_set(self):
        '''
            Sample n_samples from self.dataset and return as single tensor
        '''

        samples = []
        random.seed(0)
        indices = random.sample(range(len(self.dataset)), self.n_samples)
        
        for i in indices:
            data = self.dataset[i]
            sample = self._prepare_input(data[0])
            samples.append(sample)
            
        samples = torch.cat(samples, dim = 0)
        return samples

    # return layers of model to consider for quantization (modify with config file)
    def _get_model_layer_groups(self):
        raise NotImplementedError('_get_model_layers')
    
    # process calibration set inputs
    def _prepare_input(self):
        raise NotImplementedError('_prepare_input')
    
    # return groups of modules for weight grouping, scales calculation
    def _group_modules_for_scaling(self, layer, linear_inputs, layer_group):
        raise NotImplementedError('_group_modules_for_scaling')
    

# ======================================================================
# BLip2ForCondtionalGeneration (captioning task) AWQ Quantizer Class
# ======================================================================
class Blip2ForConditionalGenerationAWQQuantizer(BaseAWQQuantizer):

    def __init__(self, model, device, inputs_processor, dataset):
        assert isinstance(model, Blip2ForConditionalGeneration)

        super().__init__(model, device, inputs_processor, dataset)
        self.run_model = model.generate
        
    def _get_model_layer_groups(self):
        # NOTE: should ensure that keys are defined sequentially for early quitting of calibration set run
        return {'vit_layers': self.model.vision_model.encoder.layers,
                'qformer_layers': self.model.qformer.encoder.layer,
                'llm_layers': self.model.language_model.model.decoder.layers
               }

    def _prepare_input(self, inp):
        X = self.inputs_processor(images=inp, return_tensors="pt").to(self.device)
        return X['pixel_values']
    
    def _group_modules_for_scaling(self, layer, linear_inputs, layer_group):
        grouped_mods = []

        if layer_group == 'vit_layers':
            
            grouped_mods.append(
                # vit self-attn
                dict(
                    prev_op = layer.layer_norm1,
                    modules = [layer.self_attn.qkv],
                    inp = linear_inputs['self_attn.qkv'],
                    parent_module = layer.self_attn,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

            grouped_mods.append(
                # vit fc1
                dict(
                    prev_op = layer.layer_norm2,
                    modules = [layer.mlp.fc1],
                    inp = linear_inputs['mlp.fc1'],
                    parent_module = layer.mlp.fc1,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

            grouped_mods.append(
                 # vit fc2
                dict(
                    prev_op = layer.mlp.fc1,
                    modules = [layer.mlp.fc2],
                    inp = linear_inputs['mlp.fc2'],
                    parent_module = layer.mlp.fc2,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

        elif layer_group == 'qformer_layers':
            
            # Qformer self-attn QKV
            grouped_mods.append(
                dict(
                    prev_op = 'attention.attention',
                    modules = [
                        layer.attention.attention.query,
                        layer.attention.attention.key,
                        layer.attention.attention.value
                    ],
                    inp = linear_inputs['attention.attention.query'],
                    parent_module = layer.attention.attention,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

            # Qformer self-attn output
            grouped_mods.append(
                dict(
                    prev_op = 'attention.output.dense',
                    modules = [layer.attention.output.dense],
                    inp = linear_inputs['attention.output.dense'],
                    parent_module = layer.attention.output.dense,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

            # Qformer intermediate_query
            grouped_mods.append(
                dict(
                    prev_op = 'intermediate_query',
                    modules = [layer.intermediate_query.dense],
                    inp = linear_inputs['intermediate_query.dense'],
                    parent_module = layer.intermediate_query,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

            # Qformer output_query
            grouped_mods.append(
                dict(
                    prev_op = 'output_query.dense',
                    modules = [layer.output_query.dense],
                    inp = linear_inputs['output_query.dense'],
                    parent_module = layer.output_query.dense,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

            # Qformer cross-attn (only present every other layer)
            if hasattr(layer, 'crossattention'):
                #  NOTE: Qformer cross-attn QKV cant be grouped together (unlike self-attn) 
                #  because of different sizes of hidden_states and encoder_hidden_states 

                grouped_mods.append(
                    dict(
                        prev_op = 'crossattention.attention.query',
                        modules = [
                            layer.crossattention.attention.query,
                        ],
                        inp = linear_inputs['crossattention.attention.query'],
                        parent_module = layer.crossattention.attention.query,
                        layer_kwargs = self.layer_kwargs[layer_group]
                    )
                )
                grouped_mods.append(
                    dict(
                        prev_op = 'crossattention.attention.key',
                        modules = [
                            layer.crossattention.attention.key,
                        ],
                        inp = linear_inputs['crossattention.attention.key'],
                        parent_module = layer.crossattention.attention.key,
                        layer_kwargs = self.layer_kwargs[layer_group]
                    )
                )
                grouped_mods.append(
                    dict(
                        prev_op = 'crossattention.attention.value',
                        modules = [
                            layer.crossattention.attention.value
                        ],
                        inp = linear_inputs['crossattention.attention.value'],
                        parent_module = layer.crossattention.attention.value,
                        layer_kwargs = self.layer_kwargs[layer_group]
                    )
                )

                # Qformer cross-attn output
                grouped_mods.append(
                    dict(
                        prev_op = 'crossattention.output.dense',
                        modules = [layer.crossattention.output.dense],
                        inp = linear_inputs['crossattention.output.dense'],
                        parent_module = layer.crossattention.output.dense,
                        layer_kwargs = self.layer_kwargs[layer_group]
                    )
                )


        elif layer_group == 'llm_layers':
            
            assert layer.do_layer_norm_before, "llm do_layer_norm_before set to false"

            # llm attn
            grouped_mods.append(
                dict(
                    prev_op = layer.self_attn_layer_norm,
                    modules = [
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                    ],
                    inp = linear_inputs['self_attn.q_proj'],
                    parent_module = layer.self_attn,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

            # llm attn output
            grouped_mods.append(
                dict(
                    prev_op = layer.self_attn.v_proj,
                    modules = [layer.self_attn.out_proj],
                    inp = linear_inputs['self_attn.out_proj'],
                    parent_module = layer.self_attn.out_proj,
                    layer_kwargs = self.layer_kwargs[layer_group]

                )
            )

            # LLM FC1
            grouped_mods.append(
                dict(
                    prev_op = layer.final_layer_norm,
                    modules = [layer.fc1],
                    inp = linear_inputs['fc1'],
                    parent_module = layer.fc1,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

          # LLM FC2
            grouped_mods.append(
                dict(
                    prev_op = layer.fc1,
                    modules = [layer.fc2],
                    inp = linear_inputs['fc2'],
                    parent_module = layer.fc2,
                    layer_kwargs = self.layer_kwargs[layer_group]
                )
            )

        return grouped_mods
        


# ======================================================================
# BLip2ForImageTextRetrieval (retrieval task) AWQ Quantizer Class
# ======================================================================
# TODO:
class Blip2ForImageTextRetrievalAWQQuantizer(BaseAWQQuantizer):

    def __init__(self, model, device, inputs_processor, dataset):
        assert isinstance(model, Blip2ForImageTextRetrieval)
        super().__init__(model, device, inputs_processor, dataset)
        self.run_model = model.forward
        
    def _get_model_layer_groups(self):
        # NOTE: should ensure that keys are defined sequentially for early quitting of calibration set run
        return {'vit_layers': self.model.vision_model.encoder.layers,
                'qformer_layers': self.model.qformer.encoder.layer,}

    def _get_calibration_set(self):
        return [self.dataset[0], self.dataset[1]]

    def _prepare_input(self, batch):
        X = self.processor(images=batch[0], text=batch[1][0], return_tensors="pt").to(self.device, torch.float16)
        return X
