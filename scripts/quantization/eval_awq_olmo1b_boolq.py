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


def load_awq_fake_olmo1b(model_path: str, awq_path: str, w_bit: int = 4, q_group_size: int = 128):
    q_config = {"zero_point": True, "q_group_size": q_group_size}

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

    awq_results = torch.load(awq_path, map_location="cpu")
    apply_awq(model, awq_results)
    pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)

    model.eval().cuda()
    return model, tokenizer


@torch.no_grad()
def score_continuation(model, tokenizer, prompt: str, continuation: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    continuation_log_prob = token_log_probs[:, prompt_len - 1 : full_len - 1].sum()
    return continuation_log_prob.item()


def build_prompt_boolq(example: dict) -> str:
    passage = example["passage"].strip()
    question = example["question"].strip()
    return (
        "Read the passage and answer the question.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/olmo1b")
    parser.add_argument("--awq-path", default="quantization/awq/olmo1b/olmo1b-w4-g128.pt4")
    parser.add_argument("--output-json", default="results/quantization/olmo1b/awq_w4/boolq.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer = load_awq_fake_olmo1b(
        model_path=args.model_path,
        awq_path=args.awq_path,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    ds = load_dataset("boolq", split="validation")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    predictions = []
    correct = 0

    for ex in tqdm(ds, desc="Evaluating BoolQ"):
        prompt = build_prompt_boolq(ex)
        yes_score = score_continuation(model, tokenizer, prompt, " yes")
        no_score = score_continuation(model, tokenizer, prompt, " no")

        pred = "yes" if yes_score > no_score else "no"
        gold = "yes" if ex["answer"] else "no"
        is_correct = pred == gold
        correct += int(is_correct)

        predictions.append(
            {
                "question": ex["question"],
                "gold": gold,
                "prediction": pred,
                "yes_score": yes_score,
                "no_score": no_score,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(predictions) if predictions else 0.0

    result = {
        "task": "boolq",
        "model_path": args.model_path,
        "awq_path": args.awq_path,
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

    print(json.dumps(
        {
            "task": "boolq",
            "accuracy": accuracy,
            "num_correct": correct,
            "num_total": len(predictions),
            "output_json": str(out_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()