# blip2-coco
- [ ] Add vqa2 dataset+test
- [ ] Migrate datasets to HF
- [ ] Put up class for model w/ quantization
- [ ] Finish testing over ranges
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
