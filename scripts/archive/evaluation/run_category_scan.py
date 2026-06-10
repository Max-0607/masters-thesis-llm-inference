import argparse
import json
import torch

from configs.models import MODEL_CONFIGS
from src.model_loader import load_model_and_tokenizer
from src.prompts import CATEGORY_PROMPTS
from src.hooks import ActivationRecorder, get_nested_attr
from src.activation_analysis import summarize_all_layers, aggregate_category_results
from src.candidate_extraction import build_candidates, top_candidates_by_score
from src.plotting import plot_category_layer_maxima
from src.utils import ensure_dir, set_seed


def run_single_prompt(model, tokenizer, prompt, layer_path, down_proj_path):
    layers = get_nested_attr(model, layer_path)

    recorder = ActivationRecorder()
    recorder.register_on_layers(layers, down_proj_path)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    recorder.remove()

    input_summaries = summarize_all_layers(recorder.inputs)
    output_summaries = summarize_all_layers(recorder.outputs)

    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return input_summaries, output_summaries


def unique_candidates_keep_best_score(candidates):
    """
    Deduplicate candidates by (layer, row, col).
    If the same weight appears multiple times across categories,
    keep only the version with the highest score.
    """
    best_by_key = {}

    for candidate in candidates:
        key = (candidate["layer"], candidate["row"], candidate["col"])
        score = candidate["input_max_abs"] * candidate["output_max_abs"]

        candidate_copy = dict(candidate)
        candidate_copy["score"] = score

        if key not in best_by_key or score > best_by_key[key]["score"]:
            best_by_key[key] = candidate_copy

    return list(best_by_key.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/category_scan")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.output_dir)

    cfg = MODEL_CONFIGS[args.model_key]
    model, tokenizer = load_model_and_tokenizer(cfg["hf_name"])

    all_input_results = {}
    all_output_results = {}

    for category, prompts in CATEGORY_PROMPTS.items():
        per_prompt_input = []
        per_prompt_output = []

        print(f"Running category: {category}")

        for prompt in prompts:
            input_summaries, output_summaries = run_single_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layer_path=cfg["layer_path"],
                down_proj_path=cfg["down_proj_path"],
            )
            per_prompt_input.append(input_summaries)
            per_prompt_output.append(output_summaries)

        aggregated_input = aggregate_category_results(per_prompt_input)
        aggregated_output = aggregate_category_results(per_prompt_output)

        all_input_results[category] = aggregated_input
        all_output_results[category] = aggregated_output

    # Save full category scan results
    save_json = f"{args.output_dir}/{args.model_key}_category_results.json"
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": all_input_results,
                "output": all_output_results,
            },
            f,
            indent=2,
        )

    # Save plots
    plot_category_layer_maxima(
        category_to_summaries=all_input_results,
        key="max_abs_value",
        title=f"{args.model_key}: Max |down_proj input| per layer by category",
        save_path=f"{args.output_dir}/{args.model_key}_input_by_category.png",
    )

    plot_category_layer_maxima(
        category_to_summaries=all_output_results,
        key="max_abs_value",
        title=f"{args.model_key}: Max |down_proj output| per layer by category",
        save_path=f"{args.output_dir}/{args.model_key}_output_by_category.png",
    )

    # Top-K per category
    top_per_category = {}
    for category in CATEGORY_PROMPTS.keys():
        candidates = build_candidates(
            input_summaries=all_input_results[category],
            output_summaries=all_output_results[category],
        )
        unique_candidates = unique_candidates_keep_best_score(candidates)
        top_per_category[category] = sorted(
            unique_candidates,
            key=lambda x: x["score"],
            reverse=True,
        )[:args.top_k]

    save_top_per_category = f"{args.output_dir}/{args.model_key}_top{args.top_k}_per_category.json"
    with open(save_top_per_category, "w", encoding="utf-8") as f:
        json.dump(top_per_category, f, indent=2)

    # Global unique candidates across all categories
    all_candidates = []
    for category in CATEGORY_PROMPTS.keys():
        category_candidates = build_candidates(
            input_summaries=all_input_results[category],
            output_summaries=all_output_results[category],
        )

        for candidate in category_candidates:
            candidate_copy = dict(candidate)
            candidate_copy["category"] = category
            all_candidates.append(candidate_copy)

    unique_global_candidates = unique_candidates_keep_best_score(all_candidates)
    top_global = sorted(
        unique_global_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )[:args.top_k]

    save_top_global = f"{args.output_dir}/{args.model_key}_top{args.top_k}_global.json"
    with open(save_top_global, "w", encoding="utf-8") as f:
        json.dump(top_global, f, indent=2)

    print(f"Saved category results to: {save_json}")
    print(f"Saved top-{args.top_k} per category to: {save_top_per_category}")
    print(f"Saved top-{args.top_k} global candidates to: {save_top_global}")
    print("Top global unique candidates:")
    print(json.dumps(top_global, indent=2))


if __name__ == "__main__":
    main()