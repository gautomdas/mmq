import json
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import os
import sys

class ScoringPipeline:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            print("Adding current path to python system paths")
            sys.path.append(current_dir)

        self.tokenizer = PTBTokenizer()
        # self.scorers = [
        #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        #     (Meteor(), "METEOR"),
        #     (Rouge(), "ROUGE_L"),
        #     (Cider(), "CIDEr"),
        #     (Spice(), "SPICE")
        # ]
        self.scorers = [
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]

    def load_results(self, filename):
        with open(filename, 'r') as f:
            return json.load(f)

    def compute_scores(self, results, task):
        if task == 'image_captioning':
            return self._compute_image_captioning_scores(results)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _compute_image_captioning_scores(self, results):
        gts = {i: [{'caption': c} for c in ref] for i, ref in enumerate(results['references'])}
        res = {i: [{'caption': p['caption']}] for i, p in enumerate(results['predictions'])}

        print('Tokenizing...')
        gts_tokenized = self.tokenizer.tokenize(gts)
        res_tokenized = self.tokenizer.tokenize(res)

        scores = {}
        print('Computing scores...')
        for scorer, method in self.scorers:
            print(f'Computing {method} score...')
            score, scores_per_caption = scorer.compute_score(gts_tokenized, res_tokenized)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores_per_caption, method):
                    scores[m] = sc
                    scores[f'{m}_per_caption'] = scs
            else:
                scores[method] = score
                scores[f'{method}_per_caption'] = scores_per_caption

        return scores