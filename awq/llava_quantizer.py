from awq.quantizer import BaseAWQQuantizer
from awq.scaled_modules import ScaledModule
from awq.utils import *
import random

class LlavaAWQQuantizer(BaseAWQQuantizer):

    def __init__(self, model, device, inputs_processor, dataset, config, 
                 dataset_name = 'VQAv2'):
        super().__init__(model, device, inputs_processor, dataset, config)
        self.group2modules = self._get_group2modules()
        # TODO: change, testing for now
        self.n_samples = 128
        self.dataset_name = dataset_name

        # keep track of excluded modules for model size computation
        self.excluded_mods = []

    def _get_group2modules(self):
        
        group2modules = {}

        if 'vision_layers' in self.config:
            group2modules['vision_layers'] = {
                'self_attn': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj'],
                'mlp': ['mlp.fc1', 'mlp.fc2']
            } 

        if 'llm_layers' in self.config:
            group2modules['llm_layers'] = {
                'self_attn': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
                'mlp': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
            }

        return group2modules

    def _get_model_layer_groups(self):
        
        layer_groups = {}

        quant_vision_flag = 'vision_layers' in self.config
        if quant_vision_flag:
            layer_groups['vision_layers'] = self.model.vision_tower.vision_model.encoder.layers

        self.excluded_mods.extend(get_mods(self.model.vision_tower, non_linears_only=quant_vision_flag))
      

        quant_llm_flag = 'llm_layers' in self.config
        if quant_llm_flag:
            layer_groups['llm_layers'] = self.model.language_model.model.layers
        
        self.excluded_mods.extend(get_mods(self.model.language_model, non_linears_only=quant_llm_flag))

        print(f'layer_groups: {layer_groups}')
        return layer_groups

    def _group_modules_for_scaling(self, layer, linear_inputs, layer_group):
        
        grouped_mods = []

        if layer_group == 'vision_layers':
            
            if 'self_attn' in self.config[layer_group]:

                # self_attn input
                grouped_mods.append(
                    dict(
                        prev_op = layer.layer_norm1,
                        modules = [
                            layer.self_attn.k_proj,
                            layer.self_attn.q_proj,
                            layer.self_attn.v_proj,
                        ],
                        inp = linear_inputs['self_attn.q_proj'],
                        parent_module = layer.self_attn,
                        layer_kwargs = self.layer_kwargs,
                        w_bits = self.config[layer_group]['self_attn']
                    )
                )
                # self_attn output
                grouped_mods.append(
                    dict(
                        prev_op = layer.self_attn.v_proj,
                        modules = [
                            layer.self_attn.out_proj
                        ],
                        inp = linear_inputs['self_attn.out_proj'],
                        parent_module = layer.self_attn.out_proj,
                        layer_kwargs = self.layer_kwargs,
                        w_bits = self.config[layer_group]['self_attn']
                    )
                )

            if 'mlp' in self.config[layer_group]:
                 # fc1
                grouped_mods.append(
                    dict(
                        prev_op = layer.layer_norm2,
                        modules = [layer.mlp.fc1],
                        inp = linear_inputs['mlp.fc1'],
                        parent_module = layer.mlp.fc1,
                        layer_kwargs = self.layer_kwargs,
                        w_bits = self.config[layer_group]['mlp']
                    )
                )
                # fc2
                grouped_mods.append(
                    dict(
                        prev_op = layer.mlp.fc1,
                        modules = [layer.mlp.fc2],
                        inp = linear_inputs['mlp.fc2'],
                        parent_module = layer.mlp.fc2,
                        layer_kwargs = self.layer_kwargs,
                        w_bits = self.config[layer_group]['mlp']
                    )
                )

        # TODO: vision_layers, projector
        if layer_group == 'llm_layers':
            if 'self_attn' in self.config[layer_group]:
                # self_attn input
                # grouped_mods.append(
                #     dict(
                #         prev_op = layer.input_layernorm,
                #         modules = [
                #             layer.self_attn.q_proj,
                #             layer.self_attn.k_proj,
                #             layer.self_attn.v_proj,
                #         ],
                #         inp = linear_inputs['self_attn.q_proj'],
                #         parent_module = layer.self_attn,
                #         layer_kwargs = self.layer_kwargs[layer_group],
                #         w_bits = self.config[layer_group]['self_attn']
                #     )
                # )

                # self_attn query
                grouped_mods.append(
                    dict(
                        prev_op = 'self_attn.q_proj',
                        modules = [
                            layer.self_attn.q_proj
                        ],
                        inp = linear_inputs['self_attn.q_proj'],
                        parent_module = layer.self_attn.q_proj,
                        layer_kwargs = self.layer_kwargs[layer_group],
                        w_bits = self.config[layer_group]['self_attn']
                    )
                )

                # self_attn key
                grouped_mods.append(
                    dict(
                        prev_op = 'self_attn.k_proj',
                        modules = [
                            layer.self_attn.k_proj
                        ],
                        inp = linear_inputs['self_attn.k_proj'],
                        parent_module = layer.self_attn.k_proj,
                        layer_kwargs = self.layer_kwargs[layer_group],
                        w_bits = self.config[layer_group]['self_attn']
                    )
                )

                # self_attn value
                grouped_mods.append(
                    dict(
                        prev_op = 'self_attn.v_proj',
                        modules = [
                            layer.self_attn.v_proj
                        ],
                        inp = linear_inputs['self_attn.v_proj'],
                        parent_module = layer.self_attn.v_proj,
                        layer_kwargs = self.layer_kwargs[layer_group],
                        w_bits = self.config[layer_group]['self_attn']
                    )
                )

                # self_attn output
                # NOTE: sometimes skipped in the AutoAWQ implementation according to: https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
                    grouped_mods.append(
                        dict(
                            prev_op = layer.self_attn.v_proj,
                            modules = [
                                layer.self_attn.o_proj
                            ],
                            inp = linear_inputs['self_attn.o_proj'],
                            parent_module = layer.self_attn.o_proj,
                            layer_kwargs = self.layer_kwargs[layer_group],
                            w_bits = self.config[layer_group]['self_attn']
                        )
                    )

                
            if 'mlp' in self.config[layer_group]:
                # linear 1
                grouped_mods.append(
                    dict(
                        prev_op = layer.post_attention_layernorm,
                        modules = [layer.mlp.gate_proj, layer.mlp.up_proj],
                        inp = linear_inputs['mlp.gate_proj'],
                        parent_module = layer.mlp,
                        layer_kwargs = self.layer_kwargs[layer_group],
                        w_bits = self.config[layer_group]['mlp']
                    )
                )

                # linear 2
                grouped_mods.append(
                    dict(
                        prev_op = layer.mlp.up_proj,
                        modules = [layer.mlp.down_proj],
                        inp = linear_inputs['mlp.down_proj'],
                        parent_module = layer.mlp.down_proj,
                        layer_kwargs = self.layer_kwargs[layer_group],
                        w_bits = self.config[layer_group]['mlp']
                    )
                )

            

        return grouped_mods


    # NOTE: assuming VQAv2 dataset for now
    def _get_calibration_set(self):
        random.seed(self.seed)
        indices = random.sample(range(len(self.dataset)), self.n_samples)
        
        # if self.dataset_name == 'VQAv2':
        imgs = []
        prompts = []
        for i in indices:

            # short answer prompting according to: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
            # prompt = 'USER: <image>\n' + self.dataset.qa_pairs[i]['question'] + '\nAnswer the question using a single word or phrase. ASSISTANT:'
            prompt = self.dataset[i]['text_input']
            prompts.append(prompt)

            imgs.append(self.dataset[i]['image'])

        # apply inputs processor 
        samples = self.inputs_processor(images = imgs,
                                        text = prompts,
                                        return_tensors='pt',
                                        padding=True).to(self.model.device)

        return samples
        

    def _run_model(self, calibration_set):
        out = self.model.generate(**calibration_set, use_cache=False)
        out = out.to('cpu')
        clear_memory(out)


