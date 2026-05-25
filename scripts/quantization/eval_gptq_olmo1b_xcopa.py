import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from quantization.gptq.gptq_repo.eval_gptq_olmo1b_ppl import (
    get_olmo,
    get_wikitext2_olmo,
    olmo_sequential,
    DEV,
)


def score_choice(model, tokenizer, prompt, choice, device):
    full_text = prompt.strip() + " " + choice.strip()

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    if full_ids.shape[1] > model.seqlen:
        full_ids = full_ids[:, -model.seqlen:]
        prompt_ids = prompt_ids[:, -min(prompt_ids.shape[1], model.seqlen):]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        prompt_len = prompt_ids.shape[1]
        cont_log_probs = token_log_probs[:, prompt_len - 1 :]

        return cont_log_probs.sum().item()


def get_prompt(language, question):
    prompts = {
        "en": {
            "cause": "What was the cause?",
            "effect": "What happened as a result?",
        },
        "et": {
            "cause": "Mis oli põhjus?",
            "effect": "Mis juhtus selle tulemusena?",
        },
        "id": {
            "cause": "Apa penyebabnya?",
            "effect": "Apa yang terjadi sebagai hasilnya?",
        },
        "it": {
            "cause": "Qual è stata la causa?",
            "effect": "Che cosa è successo come risultato?",
        },
        "tr": {
            "cause": "Sebep neydi?",
            "effect": "Sonuç olarak ne oldu?",
        },
        "zh": {
            "cause": "原因是什么？",
            "effect": "结果发生了什么？",
        },
    }
    return prompts[language][question]


def load_examples(language, limit):
    if language == "en":
        ds = load_dataset("super_glue", "copa", split="validation")
    else:
        ds = load_dataset("xcopa", language, split="validation")

    examples = []
    for row in ds:
        examples.append({
            "premise": row["premise"],
            "question": row["question"],
            "choice1": row["choice1"],
            "choice2": row["choice2"],
            "label": int(row["label"]),
        })
        if limit is not None and len(examples) >= limit:
            break

    return examples


def evaluate_xcopa(model, tokenizer, language="en", limit=500, device="cuda:0"):
    examples = load_examples(language, limit)

    correct = 0
    results = []

    for i, ex in enumerate(examples):
        prompt = f"{ex['premise'].strip()}\n{get_prompt(language, ex['question'])}\nAnswer:"

        score1 = score_choice(model, tokenizer, prompt, ex["choice1"], device)
        score2 = score_choice(model, tokenizer, prompt, ex["choice2"], device)

        pred = 0 if score1 >= score2 else 1
        label = int(ex["label"])

        is_correct = int(pred == label)
        correct += is_correct

        results.append({
            "index": i,
            "prediction": pred,
            "label": label,
            "correct": bool(is_correct),
            "scores": [score1, score2],
            "question": ex["question"],
        })

        print(f"{i+1}/{len(examples)} | pred={pred} label={label} | acc={correct/(i+1):.4f}")

    return correct / len(examples), results


if __name__ == "__main__":
    model_id = "allenai/OLMo-1B-0724-hf"
    language = "en"
    output_json = Path("results/quantization/olmo1b/gptq/xcopa.json")

    args = {
        "nsamples": 4,
        "wbits": 4,
        "groupsize": 128,
        "percdamp": 0.01,
        "act_order": True,
    }

    limit = 500

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id}")
    model = get_olmo(model_id)
    model.eval()

    print("Loading calibration data ...")
    dataloader, _ = get_wikitext2_olmo(
        nsamples=args["nsamples"],
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print("Running GPTQ quantization ...")
    tick = time.time()
    olmo_sequential(model, dataloader, DEV, args)
    print(f"Quantization done in {time.time() - tick:.2f}s")

    model = model.to(DEV)
    model.eval()

    print(f"Evaluating XCOPA/COPA language={language} ...")
    acc, results = evaluate_xcopa(
        model,
        tokenizer,
        language=language,
        limit=limit,
        device=DEV,
    )

    out = {
        "model": "olmo-1b",
        "model_id": model_id,
        "method": "gptq_runtime",
        "task": "copa" if language == "en" else "xcopa",
        "display_task": "COPA-English" if language == "en" else "XCOPA",
        "language": language,
        "bits": args["wbits"],
        "groupsize": args["groupsize"],
        "nsamples": args["nsamples"],
        "limit": limit,
        "accuracy": acc,
        "results": results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Saved results to {output_json}")
