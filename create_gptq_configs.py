import itertools
import json
import os

bit_options = [2, 3, 4, 5, 6, 8, 16]

dir_name = "configs"
os.makedirs(dir_name, exist_ok=True)

# All 3-part combinations: vision, qformer, language
all_bits = list(itertools.product(bit_options, repeat=3))

# Shared quantization settings
base_config = {
    "percent_dampening": 0.01,
    "group_size": -1,
    "use_symmetric": True,
    "use_act_order": False,
    "use_static_groups": False,
}

for i, (vit_bits, qformer_bits, llm_bits) in enumerate(all_bits):
    json_config = {
        "vision": {"bits": vit_bits, **base_config},
        "qformer": {"bits": qformer_bits, **base_config},
        "language": {"bits": llm_bits, **base_config},
    }

    filename = os.path.join(dir_name, f"gptq_{i}.json")
    with open(filename, "w") as f:
        json.dump(json_config, f, indent=2)
