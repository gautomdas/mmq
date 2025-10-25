import sys

sys.path.append("..")
import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from scoring_pipeline import ScoringPipeline


def compute_scores(results_dir, task):
    scorer = ScoringPipeline()

    gather = []
    for results_file in tqdm(os.listdir(results_dir)):
        results_path = os.path.join(results_dir, results_file)

        with open(results_path, "r") as f:
            results = json.load(f)

            # post-processing llava output
            answers = results["answers"]
            for ans in answers:
                ans["answer"] = ans["answer"].split("ASSISTANT: ")[-1]

            if task == "vqav2":
                ann_root = "./data/vqav2/annotations"
                q_root = "./data/vqav2/questions"

                # results["answers"] = answers
                results["annotations"] = os.path.join(
                    ann_root, "v2_mscoco_val2014_annotations.json"
                )
                results["questions"] = os.path.join(
                    q_root, "v2_OpenEnded_mscoco_val2014_questions.json"
                )

                score = scorer.compute_scores(results, task)
                # print(score)

                record = dict(
                    vision_bits=results["vision_bits"],
                    language_bits=results["language_bits"],
                )

                record.update(score)

                print(record)

            elif task == "gqa":
                score = scorer.compute_scores(answers, task)["acc"]

                record = dict(
                    vision_bits=results["vision_bits"],
                    language_bits=results["language_bits"],
                    acc=score,
                )

            gather.append(record)

    return pd.DataFrame(gather)


results_dir = "./llava/gptq/vqav2"
df_vqav2_gptq = compute_scores(results_dir, "vqav2")


output_path = "./final_results/llava/gptq_vqav2.csv"
df_vqav2_gptq.to_csv(output_path, index=None)

print(f"output file written to: {output_path}")
