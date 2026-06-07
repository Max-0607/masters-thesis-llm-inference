import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quantization.awq.awq.quantize.pre_quant import apply_awq
from quantization.awq.awq.quantize.quantizer import pseudo_quantize_model_weight


SUPERWEIGHTS_OLMO1B = [
    (1, 1764, 1710),
    (1, 1764, 8041),
]


def save_superweights(model):
    sw_values = {}

    print("Saving AWQ-transformed superweights before quantization")

    for layer, row, col in SUPERWEIGHTS_OLMO1B:
        weight = model.model.layers[layer].mlp.down_proj.weight

        with torch.no_grad():
            value = weight[row, col].detach().clone().cpu()

        sw_values[(layer, row, col)] = value

        print(
            f"[SAVE SW] layer={layer}, row={row}, col={col}, "
            f"value={value.item():.6f}"
        )

    return sw_values


def restore_superweights(model, sw_values):
    print("Restoring scaled AWQ-transformed superweights after quantization")

    for (layer, row, col), value in sw_values.items():
        weight = model.model.layers[layer].mlp.down_proj.weight

        with torch.no_grad():
            old_value = weight[row, col].item()
            weight[row, col] = value.to(device=weight.device, dtype=weight.dtype)
            new_value = weight[row, col].item()

        print(
            f"[RESTORE SW] layer={layer}, row={row}, col={col}: "
            f"{old_value:.6f} -> {new_value:.6f}"
        )


def apply_uniform_superweight_scaling(model, alpha: float):
    print(f"Applying uniform superweight scaling with alpha={alpha}")

    scaling_info = {}

    for rank, (layer, row, col) in enumerate(SUPERWEIGHTS_OLMO1B):
        weight = model.model.layers[layer].mlp.down_proj.weight

        with torch.no_grad():
            old_value = weight[row, col].item()
            weight[row, col] *= alpha
            new_value = weight[row, col].item()

        scaling_info[str((layer, row, col))] = {
            "rank": rank,
            "scale": alpha,
            "old_value": old_value,
            "new_value": new_value,
        }

        print(
            f"[UNIFORM SCALE SW] rank={rank}, layer={layer}, row={row}, col={col}: "
            f"scale={alpha:.4f}, {old_value:.6f} -> {new_value:.6f}"
        )

    return scaling_info


def apply_exponential_superweight_scaling(model, alpha0: float, lambd: float):
    print(
        f"Applying exponential superweight scaling "
        f"with alpha0={alpha0}, lambda={lambd}"
    )

    scaling_info = {}

    for rank, (layer, row, col) in enumerate(SUPERWEIGHTS_OLMO1B):
        scale = alpha0 * math.exp(-lambd * rank)

        weight = model.model.layers[layer].mlp.down_proj.weight

        with torch.no_grad():
            old_value = weight[row, col].item()
            weight[row, col] *= scale
            new_value = weight[row, col].item()

        scaling_info[str((layer, row, col))] = {
            "rank": rank,
            "scale": scale,
            "old_value": old_value,
            "new_value": new_value,
        }

        print(
            f"[EXP SCALE SW] rank={rank}, layer={layer}, row={row}, col={col}: "
            f"scale={scale:.4f}, {old_value:.6f} -> {new_value:.6f}"
        )

    return scaling_info


def load_sw_awq_fake_olmo1b(
    model_path: str,
    awq_path: str,
    scaling_mode: str = "uniform",
    sw_alpha: float = 1.0,
    sw_alpha0: float = 1.0,
    sw_lambda: float = 0.0,
    w_bit: int = 4,
    q_group_size: int = 128,
):
    q_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
    }

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    print(f"Loading AWQ results from: {awq_path}")
    awq_results = torch.load(awq_path, map_location="cpu")
    apply_awq(model, awq_results)

    if scaling_mode == "uniform":
        scaling_info = apply_uniform_superweight_scaling(model, sw_alpha)
    elif scaling_mode == "exponential":
        scaling_info = apply_exponential_superweight_scaling(
            model,
            alpha0=sw_alpha0,
            lambd=sw_lambda,
        )
    else:
        raise ValueError(f"Unknown scaling_mode: {scaling_mode}")

    sw_values = save_superweights(model)

    print(
        f"Applying pseudo weight quantization: "
        f"w_bit={w_bit}, group_size={q_group_size}"
    )

    pseudo_quantize_model_weight(
        model,
        w_bit=w_bit,
        q_config=q_config,
    )

    restore_superweights(model, sw_values)

    model.eval().cuda()

    return model, tokenizer, scaling_info


@torch.no_grad()
def score_continuation(model, tokenizer, prompt: str, continuation: str) -> float:
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.cuda()

    full_ids = tokenizer(
        prompt + continuation,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.cuda()

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    outputs = model(full_ids)

    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        2,
        target_ids.unsqueeze(-1),
    ).squeeze(-1)

    continuation_log_prob = token_log_probs[
        :, prompt_len - 1 : full_len - 1
    ].sum()

    return continuation_log_prob.item()


def build_prompt_sciq(example: dict) -> str:
    support = example.get("support", "").strip()
    question = example["question"].strip()

    if support:
        return (
            "Answer the science question based on the context.\n\n"
            f"Context: {support}\n"
            f"Question: {question}\n"
            "Answer:"
        )

    return (
        "Answer the science question.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def get_sciq_choices(example: dict):
    return [
        example["correct_answer"],
        example["distractor1"],
        example["distractor2"],
        example["distractor3"],
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", default="models/olmo1b")

    parser.add_argument(
        "--awq-path",
        default="quantization/awq/olmo1b/olmo1b-w4-g128.pt4",
    )

    parser.add_argument(
        "--output-json",
        default="results/olmo1b/exp_sw_awq_sciq/sciq.json",
    )

    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)

    parser.add_argument(
        "--scaling-mode",
        choices=["uniform", "exponential"],
        default="uniform",
    )

    parser.add_argument("--sw-alpha", type=float, default=1.0)
    parser.add_argument("--sw-alpha0", type=float, default=1.0)
    parser.add_argument("--sw-lambda", type=float, default=0.0)

    args = parser.parse_args()

    model, tokenizer, scaling_info = load_sw_awq_fake_olmo1b(
        model_path=args.model_path,
        awq_path=args.awq_path,
        scaling_mode=args.scaling_mode,
        sw_alpha=args.sw_alpha,
        sw_alpha0=args.sw_alpha0,
        sw_lambda=args.sw_lambda,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    ds = load_dataset("sciq", split="validation")

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    predictions = []
    correct = 0

    for ex in tqdm(
        ds,
        desc=(
            f"Evaluating SciQ SW-AWQ "
            f"mode={args.scaling_mode}, "
            f"alpha={args.sw_alpha}, "
            f"alpha0={args.sw_alpha0}, "
            f"lambda={args.sw_lambda}"
        ),
    ):
        prompt = build_prompt_sciq(ex)
        choices = get_sciq_choices(ex)

        scores = [
            score_continuation(
                model,
                tokenizer,
                prompt,
                " " + choice.strip(),
            )
            for choice in choices
        ]

        pred_idx = int(torch.tensor(scores).argmax().item())
        gold_idx = 0

        is_correct = pred_idx == gold_idx
        correct += int(is_correct)

        predictions.append(
            {
                "question": ex["question"],
                "support": ex.get("support", ""),
                "choices": choices,
                "gold": gold_idx,
                "prediction": pred_idx,
                "scores": scores,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(predictions) if predictions else 0.0

    result = {
        "task": "sciq",
        "method": "superweight_awq",
        "scaling_mode": args.scaling_mode,
        "model_path": args.model_path,
        "awq_path": args.awq_path,
        "sw_alpha": args.sw_alpha,
        "sw_alpha0": args.sw_alpha0,
        "sw_lambda": args.sw_lambda,
        "scaling_info": scaling_info,
        "superweights": SUPERWEIGHTS_OLMO1B,
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "limit": args.limit,
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": len(predictions),
        "predictions": predictions,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "task": "sciq",
                "method": "superweight_awq",
                "scaling_mode": args.scaling_mode,
                "sw_alpha": args.sw_alpha,
                "sw_alpha0": args.sw_alpha0,
                "sw_lambda": args.sw_lambda,
                "accuracy": accuracy,
                "num_correct": correct,
                "num_total": len(predictions),
                "output_json": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()