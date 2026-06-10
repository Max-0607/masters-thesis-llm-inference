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


def score_choice(model, tokenizer, text, device):
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    if input_ids.shape[1] > model.seqlen:
        input_ids = input_ids[:, -model.seqlen:]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss = outputs.loss.item()
    return -loss


def build_sentence(sentence, option):
    return sentence.replace("_", option)


def evaluate_winogrande(model, tokenizer, limit=500, device="cuda:0"):
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    results = []

    for i, ex in enumerate(ds):
        sent1 = build_sentence(ex["sentence"], ex["option1"])
        sent2 = build_sentence(ex["sentence"], ex["option2"])

        score1 = score_choice(model, tokenizer, sent1, device)
        score2 = score_choice(model, tokenizer, sent2, device)

        pred = 1 if score1 > score2 else 2
        label = int(ex["answer"])

        is_correct = int(pred == label)
        correct += is_correct

        results.append(
            {
                "index": i,
                "prediction": pred,
                "label": label,
                "correct": bool(is_correct),
                "scores": [score1, score2],
            }
        )

        print(
            f"{i+1}/{limit} | pred={pred} label={label} | "
            f"acc={correct/(i+1):.4f}"
        )

    return correct / len(ds), results


if __name__ == "__main__":
    model_id = "allenai/OLMo-1B-0724-hf"
    output_json = Path("results/quantization/olmo1b/gptq/winogrande.json")

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

    quantizers = olmo_sequential(
        model,
        dataloader,
        DEV,
        args,
    )

    print(f"Quantization done in {time.time() - tick:.2f}s")

    print("Using GPTQ-quantized model for evaluation ...")
    model = model.to(DEV)
    model.eval()

    print("Evaluating Winogrande ...")
    acc, results = evaluate_winogrande(
        model,
        tokenizer,
        limit=limit,
        device=DEV,
    )

    out = {
        "model": "olmo-1b",
        "model_id": model_id,
        "method": "gptq_runtime",
        "task": "winogrande",
        "bits": args["wbits"],
        "groupsize": args["groupsize"],
        "nsamples": args["nsamples"],
        "limit": limit,
        "accuracy": acc,
        "results": results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )

    print(f"Saved results to {output_json}")