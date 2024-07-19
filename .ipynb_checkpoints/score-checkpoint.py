import argparse
import os
from scoring_pipeline import ScoringPipeline


def main(config_file):
    os.makedirs('./results', exist_ok=True)
    score_file = os.path.join('./results', os.path.basename(config_file))
    print(f"Reading scores from {score_file}...")

    scorer = ScoringPipeline()
    loaded_results = scorer.load_results(score_file)
    scores = scorer.compute_scores(loaded_results, task='image_captioning')
    for metric, score in scores.items():
        if not metric.endswith('_per_caption'):
            print(f"{metric}: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a given quantization config")
    parser.add_argument(
        "config", help="Path to the quantization config JSON file")
    args = parser.parse_args()

    main(args.config)
