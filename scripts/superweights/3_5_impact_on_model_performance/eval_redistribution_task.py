from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from lm_eval import evaluator, tasks
from lm_eval.base import BaseLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hooks import get_nested_attr


MODEL_SPECS = {
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(1, 2070, 7310)],
    },
    "llama-7b": {
        "hf_name": "huggyllama/llama-7b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(2, 3968, 7003)],
    },
    "llama-13b": {
        "hf_name": "huggyllama/llama-13b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(2, 2231, 2278), (2, 2231, 6939)],
    },
    "llama-30b": {
        "hf_name": "huggyllama/llama-30b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (3, 5633, 12817),
            (3, 5633, 17439),
            (10, 5633, 14386),
        ],
    },
    "llama3-8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 788, 2427),
            (1, 1384, 2427),
            (1, 4062, 2427),
        ],
    },
    "olmo1b": {
        "hf_name": "allenai/OLMo-1B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 1764, 1710),
            (1, 1764, 8041),
        ],
    },
    "olmo7b": {
        "hf_name": "allenai/OLMo-7B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 269, 7467),
            (2, 269, 8275),
            (7, 269, 453),
            (24, 269, 2300),
        ],
    },
    "phi3-mini": {
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (2, 525, 808),
            (2, 1693, 808),
            (2, 1113, 808),
            (4, 525, 2723),
            (4, 1113, 2723),
            (4, 1693, 2723),
        ],
    },
}


def to_sw_dicts(tuples):
    return [
        {"layer": layer, "row": row, "col": col}
        for layer, row, col in tuples
    ]


class LoadedHFLM(BaseLM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return next(self.model.parameters()).device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        first_device = next(self.model.parameters()).device
        with torch.no_grad():
            return self.model(inps.to(first_device)).logits

    def _model_generate(self, context, max_length, eos_token_id):
        first_device = next(self.model.parameters()).device
        return self.model.generate(
            context.to(first_device),
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False,
        )


def get_down_proj_weight(model, layer: int, layer_path: str, down_proj_path: str):
    layers = get_nested_attr(model, layer_path)
    module = get_nested_attr(layers[layer], down_proj_path)
    return module.weight


def ablate_positions(model, positions, layer_path, down_proj_path):
    with torch.no_grad():
        for pos in positions:
            weight = get_down_proj_weight(
                model=model,
                layer=pos["layer"],
                layer_path=layer_path,
                down_proj_path=down_proj_path,
            )

            old_value = None
            try:
                old_value = float(
                    weight[pos["row"], pos["col"]]
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
            except Exception:
                old_value = None

            weight[pos["row"], pos["col"]] = 0.0

            pos["old_value"] = old_value

            print(
                f"Ablated "
                f"layer={pos['layer']} "
                f"row={pos['row']} "
                f"col={pos['col']} "
                f"old={old_value}"
            )


def sample_random_positions(model, superweights, layer_path, down_proj_path, seed: int):
    rng = random.Random(seed)
    sw_set = {
        (sw["layer"], sw["row"], sw["col"])
        for sw in superweights
    }

    random_positions = []

    for sw in superweights:
        layer = sw["layer"]
        weight = get_down_proj_weight(
            model=model,
            layer=layer,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
        )

        rows, cols = weight.shape

        while True:
            row = rng.randrange(rows)
            col = rng.randrange(cols)
            key = (layer, row, col)

            if key not in sw_set and key not in {
                (p["layer"], p["row"], p["col"])
                for p in random_positions
            }:
                break

        random_positions.append(
            {
                "layer": layer,
                "row": row,
                "col": col,
            }
        )

    return random_positions


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key",
        choices=list(MODEL_SPECS.keys()),
        required=True,
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help="Model path/id. If omitted, MODEL_SPECS hf_name is used.",
    )

    parser.add_argument("--task", default="hellaswag")
    parser.add_argument("--limit", type=int, default=500)

    parser.add_argument(
        "--ablate-superweights",
        action="store_true",
    )

    parser.add_argument(
        "--ablate-random",
        action="store_true",
        help="Ablate the same number of random weights as superweights.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    parser.add_argument("--output-json", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ablate_superweights and args.ablate_random:
        raise ValueError(
            "Use only one ablation mode: --ablate-superweights or --ablate-random"
        )

    spec = MODEL_SPECS[args.model_key]

    model_path = (
        args.model_path
        if args.model_path is not None
        else spec["hf_name"]
    )

    layer_path = spec["layer_path"]
    down_proj_path = spec["down_proj_path"]
    superweights = to_sw_dicts(spec["superweights"])

    print(f"Model key: {args.model_key}")
    print(f"Loading model: {model_path}")
    print(f"Task: {args.task}")
    print(f"Limit: {args.limit}")
    print(f"Ablate superweights: {args.ablate_superweights}")
    print(f"Ablate random: {args.ablate_random}")
    print(f"Random seed: {args.random_seed}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
        device_map=(
            "auto"
            if torch.cuda.is_available()
            else None
        ),
        trust_remote_code=True,
    )

    model.eval()

    ablated_positions = []

    if args.ablate_superweights:
        ablated_positions = [dict(sw) for sw in superweights]
        ablate_positions(
            model=model,
            positions=ablated_positions,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
        )

    elif args.ablate_random:
        ablated_positions = sample_random_positions(
            model=model,
            superweights=superweights,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
            seed=args.random_seed,
        )

        ablate_positions(
            model=model,
            positions=ablated_positions,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
        )

    lm = LoadedHFLM(model=model, tokenizer=tokenizer)

    task_dict = tasks.get_task_dict([args.task])

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.limit,
        bootstrap_iters=0,
    )

    out = {
        "model_key": args.model_key,
        "model_path": model_path,
        "task": args.task,
        "limit": args.limit,
        "ablation_mode": (
            "superweights"
            if args.ablate_superweights
            else "random"
            if args.ablate_random
            else "none"
        ),
        "random_seed": args.random_seed if args.ablate_random else None,
        "layer_path": layer_path,
        "down_proj_path": down_proj_path,
        "superweights": superweights,
        "ablated_positions": ablated_positions,
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print(json.dumps(results["results"][args.task], indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()