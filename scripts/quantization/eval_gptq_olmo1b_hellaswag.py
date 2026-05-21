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
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prompt + choice, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

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
        return token_log_probs[:, prompt_len - 1 :].sum().item()


def preprocess(text):
    return text.replace(" [title]", ". ").replace("[title]", "").strip()


def evaluate_hellaswag(model, tokenizer, limit=500, device="cuda:0"):
    ds = load_dataset("hellaswag", split="validation")
    ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    results = []

    for i, ex in enumerate(ds):
        ctx_a = preprocess(ex["ctx_a"])
        ctx_b = preprocess(ex["ctx_b"])
        prompt = f"{ctx_a} {ctx_b}".strip() if ctx_b else ctx_a

        endings = [" " + e.strip() for e in ex["endings"]]
        label = int(ex["label"])

        scores = [score_choice(model, tokenizer, prompt, e, device) for e in endings]
        pred = int(torch.tensor(scores).argmax().item())

        is_correct = int(pred == label)
        correct += is_correct

        results.append({
            "index": i,
            "prediction": pred,
            "label": label,
            "correct": bool(is_correct),
            "scores": scores,
        })

        print(f"{i+1}/{limit} | pred={pred} label={label} | acc={correct/(i+1):.4f}")

    return correct / len(ds), results


if __name__ == "__main__":
    model_id = "allenai/OLMo-1B-0724-hf"
    output_json = Path("results/quantization/olmo1b/gptq/hellaswag.json")

    args = {
        "nsamples": 32,
        "wbits": 4,
        "groupsize": 128,
        "percdamp": 0.01,
        "act_order": True,
    }

    limit = 500
    output_json.parent.mkdir(parents=True, exist_ok=True)

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

    print("Evaluating HellaSwag ...")
    acc, results = evaluate_hellaswag(model, tokenizer, limit=limit, device=DEV)

    out = {
        "model": "olmo-1b",
        "model_id": model_id,
        "method": "gptq_runtime",
        "task": "hellaswag",
        "bits": args["wbits"],
        "groupsize": args["groupsize"],
        "nsamples": args["nsamples"],
        "limit": limit,
        "accuracy": acc,
        "results": results,
    }

    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved results to {output_json}")
