from typing import Dict, List


def summarize_all_layers(activations: Dict[int, dict]) -> List[dict]:
    results = []

    for layer_idx in sorted(activations.keys()):
        summary = activations[layer_idx]
        if summary is None:
            continue
        summary = dict(summary)
        summary["layer"] = layer_idx
        results.append(summary)

    return results


def aggregate_category_results(per_prompt_results: List[List[dict]]) -> List[dict]:
    if not per_prompt_results:
        return []

    num_layers = len(per_prompt_results[0])
    aggregated = []

    for layer_idx in range(num_layers):
        layer_entries = [prompt_result[layer_idx] for prompt_result in per_prompt_results]
        best = max(layer_entries, key=lambda x: x["max_abs_value"])
        aggregated.append(best)

    return aggregated