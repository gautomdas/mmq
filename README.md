# Quantization Study of BLIP2
This repo contains results, notebooks, and code related to quantizing blip2 with various configs. To get an idea of the main logic, look at the below diagram:
![image](https://github.com/user-attachments/assets/ae2b87be-339c-4a37-856c-90d93f52d39b)

## Links
- [Home](https://gautomdas.github.io/blip2-coco)
- [3D Plot](https://gautomdas.github.io/blip2-coco/3d_plot.html)
- [GitHub Repo](https://github.com/gautomdas/blip2-coco)

## To Edit and Run Repo
To create env, run, and score:
```
# conda env create -f environment.yml`
python run.py ./configs/1.json
python score.py ./results/1.json
```

*IMPORTANT:* The scoring part of this pipeline relies on the `pycocoevalcap` python submodule. To also clone this into the repo run `git clone --recurse-submodules https://github.com/gautomdas/blip2-coco` or if you already downloaded the repo and the `pycocoevalcap` folder is still empty, run `git submodule init && git submodule update`.

## To recreate the demo file
1. Download the coco data set to the data folder using the following script (assumes you have the environment loaded): `python download_coco.py`
2. From there you should be able to run all of `demo.ipynb`
3. `demo.ipynb` goes over the 3 main steps in the diagram above

The following files are as follows:
- `run.py`: The singular file used for quantization + inferencing. This takes in a config as `./configs/<#>.json` and runs it.
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

## Running TODO
- [ ] Add vqa2 dataset+test
- [ ] Migrate datasets to HF
- [ ] Look at error propagation through layers for quantizing
- [ ] Add GPTQ and AWQ

## Interesting Results

1082.json:
```
{
  "predictions": [
    {
      "image_id": 397133,
      "caption": "the new xiaomi mi box"
    },
    {
      "image_id": 37777,
      "caption": "a white and black image of a smartphone"
    },
    {
      "image_id": 252219,
      "caption": "a white and blue box with a black and white logo"
    },
    {
      "image_id": 87038,
      "caption": "a white and black table with a white and black table cloth"
    },
    {
      "image_id": 174482,
      "caption": "an image of a white table with a black and white image"
    },
    {
      "image_id": 403385,
      "caption": "an image of a white wall with a black and white image of a speaker"
    },
    {
      "image_id": 6818,
      "caption": "the new apple tv 4k"
    },
    {
      "image_id": 480985,
      "caption": "a white and black image of a computer screen"
    },
    {
      "image_id": 458054,
      "caption": "a white and black square with a white and black square"
    },
	...
}
```
