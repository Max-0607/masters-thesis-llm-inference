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
    return parser.parse_args()


def get_token_probability(model, tokenizer, prompt, token_str, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    token_ids = tokenizer.encode(token_str, add_special_tokens=False)

    if len(token_ids) != 1:
        return {
            "token": token_str,
            "token_ids": token_ids,
            "probability": None,
            "note": "Token is not a single tokenizer token."
        }

    token_id = token_ids[0]
    prob = probs[token_id].item()

    return {
        "token": token_str,
        "token_ids": token_ids,
        "probability": prob,
        "note": ""
    }


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

    stopwords = [
    # punctuation
    ".", ",", ";", ":", "!", "?",

    # articles / determiners
    " the", " a", " an", " this", " that", " these", " those",

    # conjunctions
    " and", " or", " but", " so", " yet",

    # prepositions
    " of", " in", " to", " for", " with", " on", " at", " by", " from",

    # pronouns / function
    " I", " you", " he", " she", " they", " we", " it",
    " is", " was", " are", " were", " be", " been", " being",
    " not", " no", " yes",

    # content words
    " summer", " winter", " hot", " cold", " warm",
    " weather", " season", " temperature", " day", " night"
]
    results = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "results": []
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

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()