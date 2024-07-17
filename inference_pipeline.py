import torch
from tqdm import tqdm
import json

class InferencePipeline:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def run_inference(self, dataset, task, max_samples=None):
        if task == 'image_captioning':
            return self._run_image_captioning(dataset, max_samples)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _run_image_captioning(self, dataset, max_samples):
        results = []
        references = []
        
        for i in tqdm(range(min(len(dataset), max_samples or len(dataset)))):
            image = dataset[i][0]
            captions = dataset[i][1]
            img_id = dataset.ids[i]
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
            
            results.append({"image_id": img_id, "caption": caption})
            references.append(captions)

        return {
            'predictions': results,
            'references': references
        }

    def save_results(self, results, filename):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)