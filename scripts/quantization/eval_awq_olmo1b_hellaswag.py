import argparse
import json
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


def load_awq_fake_olmo1b(model_path, awq_path, w_bit=4, q_group_size=128):
    q_config = {"zero_point": True, "q_group_size": q_group_size}

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
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

    awq_results = torch.load(awq_path, map_location="cpu")
    apply_awq(model, awq_results)
    pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)

    model.eval().cuda()
    return model, tokenizer


@torch.no_grad()
def score_continuation(model, tokenizer, prompt, continuation):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs[:, prompt_len - 1 : full_len - 1].sum().item()


def preprocess_hellaswag_text(text):
    return text.replace(" [title]", ". ").replace("[title]", "").strip()


def build_prompt_hellaswag(example):
    ctx_a = preprocess_hellaswag_text(example["ctx_a"])
    ctx_b = preprocess_hellaswag_text(example["ctx_b"])
    return f"{ctx_a} {ctx_b}".strip() if ctx_b else ctx_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/olmo1b")
    parser.add_argument("--awq-path", default="quantization/awq/olmo1b/olmo1b-w4-g128.pt4")
    parser.add_argument("--output-json", default="results/olmo1b/awq/hellaswag_500.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer = load_awq_fake_olmo1b(
        model_path=args.model_path,
        awq_path=args.awq_path,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    ds = load_dataset("hellaswag", split="validation")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    predictions = []
    correct = 0

    for ex in tqdm(ds, desc="Evaluating HellaSwag AWQ"):
        prompt = build_prompt_hellaswag(ex)
        endings = ex["endings"]

        scores = [
            score_continuation(model, tokenizer, prompt, " " + ending.strip())
            for ending in endings
        ]

        pred_idx = int(torch.tensor(scores).argmax().item())
        gold_idx = int(ex["label"])
        is_correct = pred_idx == gold_idx
        correct += int(is_correct)

        predictions.append(
            {
                "activity_label": ex.get("activity_label", None),
                "context": prompt,
                "gold": gold_idx,
                "prediction": pred_idx,
                "scores": scores,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(predictions) if predictions else 0.0

    result = {
        "task": "hellaswag",
        "method": "awq",
        "model_path": args.model_path,
        "awq_path": args.awq_path,
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
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "task": "hellaswag",
        "method": "awq",
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": len(predictions),
        "output_json": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()