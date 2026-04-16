import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from modelutils import find_layers
from quant import Quant3Linear, make_quant3

from eval_gptq_olmo1b_ppl import get_olmo, get_wikitext2_olmo, olmo_sequential, DEV
from transformers import AutoTokenizer


def apply_quantizers(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])

    print("Packing quantized layers ...")
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print("Done.")
    return model


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
        cont_log_probs = token_log_probs[:, prompt_len - 1 :]
        return cont_log_probs.sum().item()


def evaluate_piqa(model, tokenizer, limit=20, device="cuda:0"):
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    results = []

    for i, ex in enumerate(ds):
        prompt = f"Question: {ex['goal']}\nAnswer:"
        choices = [f" {ex['sol1']}", f" {ex['sol2']}"]
        label = int(ex["label"])

        scores = [score_choice(model, tokenizer, prompt, c, device) for c in choices]
        pred = 0 if scores[0] > scores[1] else 1
        is_correct = int(pred == label)
        correct += is_correct

        results.append(
            {
                "index": i,
                "prediction": pred,
                "label": label,
                "correct": bool(is_correct),
                "scores": scores,
            }
        )

        print(f"{i+1}/{limit} | pred={pred} label={label} | acc={correct/(i+1):.4f}")

    return correct / len(ds), results


if __name__ == "__main__":
    model_id = "allenai/OLMo-1B-0724-hf"
    output_json = Path("gptq_olmo1b_piqa.json")

    args = {
        "nsamples": 4,
        "wbits": 4,
        "groupsize": 128,
        "percdamp": 0.01,
        "act_order": True,
    }
    limit = 20

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
    quantizers = olmo_sequential(model, dataloader, DEV, args)
    print(f"Quantization done in {time.time() - tick:.2f}s")

    print("Using GPTQ-quantized model for evaluation ...")
    model = model.to(DEV)
    model.eval()

    print("Evaluating PIQA ...")
    acc, results = evaluate_piqa(model, tokenizer, limit=limit, device=DEV)

    out = {
        "model": "olmo-1b",
        "model_id": model_id,
        "method": "gptq_runtime",
        "task": "piqa",
        "bits": args["wbits"],
        "groupsize": args["groupsize"],
        "nsamples": args["nsamples"],
        "limit": limit,
        "accuracy": acc,
        "results": results,
    }

    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved results to {output_json}")