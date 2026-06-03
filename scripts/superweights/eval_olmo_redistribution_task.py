from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lm_eval import evaluator, tasks
from lm_eval.base import BaseLM
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPERWEIGHTS = {
    "olmo1b": [
        {"layer": 1, "row": 1764, "col": 1710},
        {"layer": 1, "row": 1764, "col": 8041},
    ],
    "olmo7b": [
        {"layer": 1, "row": 269, "col": 7467},
        {"layer": 2, "row": 269, "col": 8275},
        {"layer": 7, "row": 269, "col": 453},
        {"layer": 24, "row": 269, "col": 2300},
    ],
}


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
        return self.tokenizer.encode(
            string,
            add_special_tokens=False,
        )

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        first_device = next(self.model.parameters()).device
        with torch.no_grad():
            return self.model(
                inps.to(first_device)
            ).logits

    def _model_generate(
        self,
        context,
        max_length,
        eos_token_id,
    ):
        first_device = next(self.model.parameters()).device
        return self.model.generate(
            context.to(first_device),
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False,
        )


def ablate_superweights(model, superweights):
    with torch.no_grad():
        for sw in superweights:
            weight = (
                model.model.layers[sw["layer"]]
                .mlp.down_proj.weight
            )

            weight[
                sw["row"],
                sw["col"]
            ] = 0.0

            print(
                f"Ablated "
                f"layer={sw['layer']} "
                f"row={sw['row']} "
                f"col={sw['col']}"
            )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        choices=["olmo1b", "olmo7b"],
        required=True,
    )

    parser.add_argument(
        "--model-path",
        required=True,
    )

    parser.add_argument(
        "--task",
        default="hellaswag",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--ablate-superweights",
        action="store_true",
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    superweights = SUPERWEIGHTS[
        args.model_type
    ]

    print(f"Model type: {args.model_type}")
    print(f"Loading model: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
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

    if args.ablate_superweights:
        ablate_superweights(
            model,
            superweights,
        )

    lm = LoadedHFLM(
        model=model,
        tokenizer=tokenizer,
    )

    task_dict = tasks.get_task_dict(
        [args.task]
    )

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.limit,
        bootstrap_iters=0,
    )

    out = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "task": args.task,
        "limit": args.limit,
        "ablate_superweights": (
            args.ablate_superweights
        ),
        "superweights": superweights,
        "results": results,
    }

    output_path = Path(
        args.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print(
        json.dumps(
            results["results"][args.task],
            indent=2,
        )
    )

    print(
        f"\nSaved to {output_path}"
    )


if __name__ == "__main__":
    main()