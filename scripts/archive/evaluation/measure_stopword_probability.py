import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-sw", action="store_true")
    return parser.parse_args()


def get_token_probability(model, tokenizer, prompt, token_str, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)

    token_ids = tokenizer.encode(token_str, add_special_tokens=False)

    if len(token_ids) != 1:
        return {
            "token": token_str,
            "token_ids": token_ids,
            "probability": None,
            "note": "Token is not a single tokenizer token.",
        }

    token_id = token_ids[0]

    return {
        "token": token_str,
        "token_ids": token_ids,
        "probability": probs[token_id].item(),
        "note": "",
    }


def ablate_olmo_superweights(model):
    with torch.no_grad():
        old_1 = model.model.layers[1].mlp.down_proj.weight[1764, 1710].item()
        old_2 = model.model.layers[2].mlp.down_proj.weight[1764, 8041].item()

        model.model.layers[1].mlp.down_proj.weight[1764, 1710] = 0.0
        model.model.layers[2].mlp.down_proj.weight[1764, 8041] = 0.0

    print(f"Applied OLMo superweight ablation:")
    print(f"  layer 1 weight[1764, 1710]: {old_1:.6f} -> 0")
    print(f"  layer 2 weight[1764, 8041]: {old_2:.6f} -> 0")


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    use_cuda = "cuda" in args.device and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if args.device != "cpu":
        model = model.to(args.device)

    model.eval()

    if args.no_sw:
        ablate_olmo_superweights(model)

    stopwords = [
    # punctuation (n=10)
    ".", ",", ";", ":", "!", "?", "-", ")", "(", "\"",

    # articles / determiners (n=10)
    " the", " a", " an", " this", " that", " these", " those",
    " some", " any", " every",

    # conjunctions (n=10)
    " and", " or", " but", " so", " yet", " because", " although",
    " while", " since", " if",

    # prepositions (n=10)
    " of", " in", " to", " for", " with", " on", " at", " by", " from", " into",

    # pronouns / function (n=10)
    " I", " you", " he", " she", " they", " we", " it",
    " is", " was", " not",

    # content words (n=10)
    " summer", " winter", " hot", " cold", " warm",
    " weather", " climate", " temperature", " day", " night",
]

    results = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "mode": "no_sw" if args.no_sw else "original",
        "results": [],
    }

    for token_str in stopwords:
        item = get_token_probability(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            token_str=token_str,
            device=args.device,
        )
        results["results"].append(item)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid = sum(1 for r in results["results"] if r["probability"] is not None)
    print(f"Saved to {args.output_path}")
    print(f"Valid single-token entries: {valid}/{len(stopwords)}")


if __name__ == "__main__":
    main()