import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from configs.superweights import SUPERWEIGHTS
from quantization.awq.awq.quantize.pre_quant import apply_awq
from quantization.awq.awq.quantize.quantizer import pseudo_quantize_model_weight


def get_superweights(model_key: str):
    if model_key not in SUPERWEIGHTS:
        raise ValueError(
            f"No superweights found for model_key='{model_key}'. "
            f"Available keys: {list(SUPERWEIGHTS.keys())}"
        )

    return [
        (item["layer"], item["row"], item["col"])
        for item in SUPERWEIGHTS[model_key]
    ]


def get_down_proj_weight(model, layer: int):
    return model.model.layers[layer].mlp.down_proj.weight


def save_superweights(model, superweights):
    sw_values = {}

    print("Saving AWQ-transformed superweights before quantization")

    for layer, row, col in superweights:
        weight = get_down_proj_weight(model, layer)

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
        weight = get_down_proj_weight(model, layer)

        with torch.no_grad():
            old_value = weight[row, col].item()
            weight[row, col] = value.to(device=weight.device, dtype=weight.dtype)
            new_value = weight[row, col].item()

        print(
            f"[RESTORE SW] layer={layer}, row={row}, col={col}: "
            f"{old_value:.6f} -> {new_value:.6f}"
        )


def apply_uniform_superweight_scaling(model, superweights, alpha: float):
    print(f"Applying uniform superweight scaling with alpha={alpha}")

    scaling_info = {}

    for rank, (layer, row, col) in enumerate(superweights):
        weight = get_down_proj_weight(model, layer)

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


def apply_exponential_superweight_scaling(model, superweights, alpha0: float, lambd: float):
    print(
        f"Applying exponential superweight scaling "
        f"with alpha0={alpha0}, lambda={lambd}"
    )

    scaling_info = {}

    for rank, (layer, row, col) in enumerate(superweights):
        scale = alpha0 * math.exp(-lambd * rank)
        weight = get_down_proj_weight(model, layer)

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


def load_sw_awq_fake_model(
    model_path: str,
    awq_path: str,
    superweights,
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
        scaling_info = apply_uniform_superweight_scaling(
            model=model,
            superweights=superweights,
            alpha=sw_alpha,
        )
    elif scaling_mode == "exponential":
        scaling_info = apply_exponential_superweight_scaling(
            model=model,
            superweights=superweights,
            alpha0=sw_alpha0,
            lambd=sw_lambda,
        )
    else:
        raise ValueError(f"Unknown scaling_mode: {scaling_mode}")

    sw_values = save_superweights(model, superweights)

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


def build_prompt_piqa(example: dict) -> str:
    return (
        "Choose the more plausible solution to the goal.\n\n"
        f"Goal: {example['goal'].strip()}\n"
        "Solution:"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--awq-path", required=True)
    parser.add_argument("--output-json", required=True)

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

    superweights = get_superweights(args.model_key)

    print(f"Using model_key={args.model_key}")
    print(f"Using superweights={superweights}")

    model, tokenizer, scaling_info = load_sw_awq_fake_model(
        model_path=args.model_path,
        awq_path=args.awq_path,
        superweights=superweights,
        scaling_mode=args.scaling_mode,
        sw_alpha=args.sw_alpha,
        sw_alpha0=args.sw_alpha0,
        sw_lambda=args.sw_lambda,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    ds = load_dataset("piqa", split="validation")

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    predictions = []
    correct = 0

    for ex in tqdm(
        ds,
        desc=(
            f"Evaluating PIQA SW-AWQ "
            f"model={args.model_key}, "
            f"mode={args.scaling_mode}, "
            f"alpha={args.sw_alpha}, "
            f"alpha0={args.sw_alpha0}, "
            f"lambda={args.sw_lambda}"
        ),
    ):
        prompt = build_prompt_piqa(ex)

        score1 = score_continuation(
            model,
            tokenizer,
            prompt,
            " " + ex["sol1"].strip(),
        )

        score2 = score_continuation(
            model,
            tokenizer,
            prompt,
            " " + ex["sol2"].strip(),
        )

        pred = "1" if score1 > score2 else "2"
        gold = "1" if int(ex["label"]) == 0 else "2"

        is_correct = pred == gold
        correct += int(is_correct)

        predictions.append(
            {
                "goal": ex["goal"],
                "gold": gold,
                "prediction": pred,
                "score1": score1,
                "score2": score2,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(predictions) if predictions else 0.0

    result = {
        "task": "piqa",
        "method": "superweight_awq",
        "model_key": args.model_key,
        "model_path": args.model_path,
        "awq_path": args.awq_path,
        "scaling_mode": args.scaling_mode,
        "sw_alpha": args.sw_alpha,
        "sw_alpha0": args.sw_alpha0,
        "sw_lambda": args.sw_lambda,
        "scaling_info": scaling_info,
        "superweights": superweights,
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
                "task": "piqa",
                "method": "superweight_awq",
                "model_key": args.model_key,
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