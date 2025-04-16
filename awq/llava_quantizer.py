from awq.quantizer import BaseAWQQuantizer
from awq.scaled_modules import ScaledModule
from awq.utils import *

class LlavaAWQQuantizer(BaseAWQQuantizer):

    def __init__(self, model, device, inputs_processor, dataset, config):
        super().__init__(model, device, inputs_processor, dataset, config)
        self.group2modules = self._get_group2modules()

        self.n_samples = 128

        # keep track of excluded modules for model size computation
        self.excluded_mods = []

    def _get_group2modules():
        pass

    def _get_model_layer_groups(self):
        
        layer_groups = {}

        quant_vision_flag = 'vision_layers' in self.config
        if quant_vision_flag:
            layer_groups['vision_layers'] = self.model.vision_tower.vision_model.encoder.layers

        self.excluded_mods.extend(get_mods(self.model.vision_tower, non_linears_only=quant_vision_flag))
      

        # TODO: multi_modal projector?

        quant_llm_flag = 'llm_layers' in self.config
        if quant_llm_flag:
            layer_groups['llm_layers'] = self.model.language_model.model.layers
        
        self.excluded_mods.extend(get_mods(self.model.language_model, non_linears_only=quant_llm_flag))

        


