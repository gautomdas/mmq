import torch
from tqdm import tqdm
import json
import numpy as np
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider

class EvaluationPipeline:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.tokenizer = PTBTokenizer()
        self.cider_scorer = Cider()

    def evaluate(self, dataset, task, max_samples=None):
        if task == 'image_captioning':
            return self._evaluate_image_captioning(dataset, max_samples)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _evaluate_image_captioning(self, dataset, max_samples):
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

        score, scores = self._compute_cider(results, references)

        return {
            'overall_cider': float(score),  # Convert to float
            'individual_cider': scores.tolist(),  # Convert to list
            'predictions': results,
            'references': references
        }

    def _compute_cider(self, predictions, references):
        gts = {i: [{'caption': c} for c in refs] for i, refs in enumerate(references)}
        res = {i: [{'caption': p['caption']}] for i, p in enumerate(predictions)}

        gts_tokenized = self.tokenizer.tokenize(gts)
        res_tokenized = self.tokenizer.tokenize(res)

        score, scores = self.cider_scorer.compute_score(gts_tokenized, res_tokenized)

        return score, scores

    def save_results(self, results, filename):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)