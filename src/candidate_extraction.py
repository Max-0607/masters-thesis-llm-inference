from typing import List, Dict


def build_candidates(input_summaries: List[Dict], output_summaries: List[Dict]) -> List[Dict]:
    """
    Build one candidate per layer:
    - output spike channel -> row
    - input spike channel -> col
    """
    input_by_layer = {x["layer"]: x for x in input_summaries}
    output_by_layer = {x["layer"]: x for x in output_summaries}

    candidates = []
    common_layers = sorted(set(input_by_layer.keys()) & set(output_by_layer.keys()))

    for layer in common_layers:
        in_sum = input_by_layer[layer]
        out_sum = output_by_layer[layer]

        candidates.append({
            "layer": layer,
            "row": out_sum["channel_idx"],
            "col": in_sum["channel_idx"],
            "input_max_abs": in_sum["max_abs_value"],
            "output_max_abs": out_sum["max_abs_value"],
        })

    return candidates


def top_candidates_by_score(candidates: List[Dict], top_k: int = 10) -> List[Dict]:
    def score(candidate: Dict) -> float:
        return candidate["input_max_abs"] * candidate["output_max_abs"]

    return sorted(candidates, key=score, reverse=True)[:top_k]