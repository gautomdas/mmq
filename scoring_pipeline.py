import json
import numpy as np
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

    def compute_scores(self, results, task, **kwargs):
        if task == 'image_captioning':
            return self._compute_image_captioning_scores(results)
        elif task == "image_text_retrieval":
            return self._compute_retrieval_scores(results, **kwargs)
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

    def _compute_retrieval_scores(self, results): 
        scores_i2t = results["scores_i2t"]
        scores_t2i = results["scores_t2i"]
        txt2img = results["txt2img"]
        img2txt = results["img2txt"]
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        return eval_result