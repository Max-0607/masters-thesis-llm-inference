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

    print(f"Applying pseudo weight quantization: w_bit={w_bit}, group_size={q_group_size}")
    pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)

    model.eval().cuda()
    return model, tokenizer


@torch.no_grad()
def score_continuation(model, tokenizer, prompt, continuation, max_length=256):
    prompt = prompt.strip()
    continuation = continuation.strip()
    full_text = prompt + " " + continuation

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    ).input_ids.cuda()

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    ).input_ids.cuda()

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        return float("-inf")

    outputs = model(
        input_ids=full_ids,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(
        2,
        target_ids.unsqueeze(-1),
    ).squeeze(-1)

    cont_start = max(prompt_len - 1, 0)
    cont_log_probs = token_log_probs[:, cont_start:]

    if cont_log_probs.numel() == 0:
        return float("-inf")

    return cont_log_probs.sum().item()


def get_language_prompt(language, question):
    prompts = {
        "en": {
            "cause": "What was the cause?",
            "effect": "What happened as a result?",
        },
        "et": {
            "cause": "Mis oli põhjus?",
            "effect": "Mis juhtus selle tulemusena?",
        },
        "ht": {
            "cause": "Ki sa ki te kòz la?",
            "effect": "Kisa ki te rive kòm rezilta?",
        },
        "id": {
            "cause": "Apa penyebabnya?",
            "effect": "Apa yang terjadi sebagai hasilnya?",
        },
        "it": {
            "cause": "Qual è stata la causa?",
            "effect": "Che cosa è successo come risultato?",
        },
        "qu": {
            "cause": "Imataq karqan?",
            "effect": "Imataq chaymanta pasaran?",
        },
        "sw": {
            "cause": "Sababu ilikuwa nini?",
            "effect": "Nini kilitokea kama matokeo?",
        },
        "ta": {
            "cause": "காரணம் என்ன?",
            "effect": "இதன் விளைவாக என்ன நடந்தது?",
        },
        "th": {
            "cause": "สาเหตุคืออะไร?",
            "effect": "เกิดอะไรขึ้นเป็นผลลัพธ์?",
        },
        "tr": {
            "cause": "Sebep neydi?",
            "effect": "Sonuç olarak ne oldu?",
        },
        "vi": {
            "cause": "Nguyên nhân là gì?",
            "effect": "Kết quả là gì?",
        },
        "zh": {
            "cause": "原因是什么？",
            "effect": "结果发生了什么？",
        },
    }

    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")

    return prompts[language][question]


def build_prompt_xcopa(example, language):
    premise = example["premise"].strip()
    question_text = get_language_prompt(language, example["question"])
    return f"{premise}\n{question_text}\nAnswer:"


def load_xcopa_or_copa(language, limit):
    examples = []

    if language == "en":
        ds = load_dataset("super_glue", "copa", split="validation")
    else:
        ds = load_dataset("xcopa", language, split="validation")

    for row in ds:
        examples.append(
            {
                "premise": row["premise"],
                "question": row["question"],
                "choice1": row["choice1"],
                "choice2": row["choice2"],
                "label": int(row["label"]),
            }
        )

        if limit is not None and len(examples) >= limit:
            break

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/olmo1b")
    parser.add_argument("--awq-path", default="quantization/awq/olmo1b/olmo1b-w4-g128.pt4")
    parser.add_argument("--output-json", default="results/quantization/olmo1b/awq_w4/xcopa.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"],
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer = load_awq_fake_olmo1b(
        model_path=args.model_path,
        awq_path=args.awq_path,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    examples = load_xcopa_or_copa(
        language=args.language,
        limit=args.limit,
    )

    predictions = []
    correct = 0

    for i, ex in enumerate(tqdm(examples, desc=f"Evaluating XCOPA/COPA-{args.language} AWQ")):
        prompt = build_prompt_xcopa(ex, args.language)

        score1 = score_continuation(
            model,
            tokenizer,
            prompt,
            ex["choice1"],
            max_length=args.max_length,
        )

        score2 = score_continuation(
            model,
            tokenizer,
            prompt,
            ex["choice2"],
            max_length=args.max_length,
        )

        pred_idx = 0 if score1 >= score2 else 1
        gold_idx = int(ex["label"])

        is_correct = pred_idx == gold_idx
        correct += int(is_correct)

        predictions.append(
            {
                "id": i,
                "premise": ex["premise"],
                "question": ex["question"],
                "choice1": ex["choice1"],
                "choice2": ex["choice2"],
                "gold": gold_idx,
                "prediction": pred_idx,
                "score_choice1": score1,
                "score_choice2": score2,
                "margin": score1 - score2,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(predictions) if predictions else 0.0

    result = {
        "task": "copa" if args.language == "en" else "xcopa",
        "display_task": "XCOPA" if args.language != "en" else "COPA-English",
        "language": args.language,
        "method": "awq",
        "model_path": args.model_path,
        "awq_path": args.awq_path,
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "limit": args.limit,
        "max_length": args.max_length,
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": len(predictions),
        "predictions": predictions,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "task": result["task"],
        "display_task": result["display_task"],
        "language": args.language,
        "method": "awq",
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": len(predictions),
        "output_json": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()