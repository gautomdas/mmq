# mmq project
This repo contains results, notebooks, and code related to quantizing blip2 with various configs. To get an idea of the main logic, look at the below diagram:
![image](https://github.com/user-attachments/assets/ae2b87be-339c-4a37-856c-90d93f52d39b)

## Links
- [Home](https://gautomdas.github.io/mmq)
- [3D Plot](https://gautomdas.github.io/mmq/plots/3d_plot.html)
- [GitHub Repo](https://github.com/gautomdas/mmq)

## Installing Dependencies
Install dependencies for BLIP-2 tasks:
```
pip3 install -r requirements_blip2.txt
```
Install dependencies for LLAVA tasks:
```
pip3 install r requirements_llava.txt
```
*IMPORTANT:* The scoring part of this pipeline relies on the `pycocoevalcap` python submodule. To also clone this into the repo run `git clone --recurse-submodules https://github.com/gautomdas/blip2-coco` or if you already downloaded the repo and the `pycocoevalcap` folder is still empty, run `git submodule init && git submodule update`.
## Installing Datasets
COCO:
```
python3 download_coco.py
```
Flickr30k (1K test set):
```
python3 download_flickr.py
```
VQAv2:
```
python3 download_vqav2.py
```
GQA:
```
python3 download_gqa.py
```
## Running Evaluations
The run scripts generally follow this structure:
```bash
python3 run_[quantization_method].py --task <task_name> --config <path_to_config.json>

| Argument        | Type         | Default     | Description |
|-----------------|--------------|-------------|-------------|
| `--distributed` | flag         | False       | Whether to use distributed inference in a single node (default: False); only supported for image captioning and VQA tasks. 
| `--batch_size`  | int          | 64          | Batch size used during inference. 
| `--num_workers` | int          | 1           | Number of worker threads for dataset loading. 
| `--task`        | string       | —           | Task to run.
| `--config`      | string       | —           | Path to the quantization config JSON file. 
| `--max_samples` | int or None  | None        | If set, restricts evaluation to the first n samples of the dataset. 
| `--dataset_dir` | string       | `./data`    | Path to the dataset directory. 
| `--output_dir`  | string       | `./output`  | Directory where results will be saved.
```
### Available Tasks
Uniform Quantization:
```
blip2-image_captioning
blip2-image_retrieval
```
GPTQ:
```
blip2-image_captioning
blip2-image_text_retrieval
blip2-vqav2
blip2-gqa
llava-vqav2
llava-gqa
```
AWQ:
```
blip2-image_captioning
blip2-image_text_retrieval
blip2-vqav2
blip2-gqa
llava-vqav2
llava-gqa
```

## To Recreate the Demo File
1. Download the coco data set to the data folder using the following script (assumes you have the environment loaded): `python download_coco.py`
2. From there you should be able to run all of `demo.ipynb`
3. `demo.ipynb` goes over the 3 main steps in the diagram above

The following files are as follows:
- `blip_quantizer.py`: The quantization class that quantizes a the blip2 model.
- `inference_pipeline.py`: The inference class that takes a model and tasks to produce `results/<#>.json`.
- `scoring_pipeline.py`: The scoring class used to convert results to scores based on task. This is separate from the inferencer/quantizer because it only requires the CPU to run.
- `quant_functions.py`: Functions that are `Tensor`->`Tensor` and perform quantization.
- `utils.py`: Additional utils used for config loading and model printing.
- `multi_sbatch.py`: Runs the `main.py` script over many GPUs and different configs.

## Notebooks
- `demo.ipynb`: The above figure demonstrated in a ipynb
- `blip2_analysis.ipynb`: Counting linear layers and params for the BLIP2 model
- `blip2_dropoff_coco.ipynb`: A look at drop off between different quantizations over the whole model
- `dataset_usage.ipynb`: A simple file showing how the COCO dataset (and others) are loaded
- `config_creator.ipynb`: Create all combinations of configs based on:
```
for each bit width:
  for each model part (ViT, LLM, QFormer):
    for each of the 8 combinations of front/middle/end:
      try with 2 other models quantized, not quantized, 1 of each, and 1 of each the other way
```

