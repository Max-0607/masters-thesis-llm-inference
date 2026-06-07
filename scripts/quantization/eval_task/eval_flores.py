import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.models import MODEL_CONFIGS
from configs.superweights import SUPERWEIGHTS
from src.hooks import get_nested_attr
from src.quantization import ActivationQuantHook


LANG_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
}


def resolve_torch_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_quant_hook(model, model_key, mode, bits):
    if mode == "fp16":
        return None

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])

    if mode == "naive":
        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=list(range(len(layers))),
            n_bits=bits,
            mode="naive",
        )

    if mode == "super":
        if model_key not in SUPERWEIGHTS:
            raise ValueError(f"No superweights registered for model_key='{model_key}'")

        sw_layers = sorted({int(e["layer"]) for e in SUPERWEIGHTS[model_key]})
        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=sw_layers,
            n_bits=bits,
            mode="super",
        )

    raise ValueError(f"Unsupported mode: {mode}")


def load_flores_examples(src, tgt, limit):
    ds = load_dataset("yash9439/flores200", split="devtest")

    print("FLORES columns loaded")

    src_col = LANG_MAP[src]
    tgt_col = LANG_MAP[tgt]

    examples = []
    for row in ds:
        src_text = row.get(src_col)
        tgt_text = row.get(tgt_col)

        if src_text is None or tgt_text is None:
            continue

        examples.append(
            {
                "src": str(src_text).strip(),
                "tgt": str(tgt_text).strip(),
            }
        )

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError(f"No valid FLORES examples found for {src}->{tgt}")

    return examples


def build_prompt(src_text, src, tgt):
    return (
        f"Translate the following sentence from {src} to {tgt}.\n"
        "Return only the translation.\n"
        "Do not explain.\n"
        "Do not add comments.\n\n"
        f"Sentence: {src_text}\n"
        "Translation:"
    )


def evaluate(model, tokenizer, examples, src, tgt):
    device = next(model.parameters()).device

    predictions = []
    losses = []

    for ex in examples:
        prompt = build_prompt(ex["src"], src, tgt)
        gold = ex["tgt"]

        full_text = prompt + " " + gold

        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        labels = inputs["input_ids"].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)

        loss = outputs.loss.detach().float().item()
        losses.append(loss)

        predictions.append(
            {
                "src": ex["src"],
                "gold": gold,
                "loss": loss,
            }
        )

    mean_loss = sum(losses) / len(losses)
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())

    return {
        "chrf": None,
        "loss": mean_loss,
        "ppl": ppl,
        "num_examples": len(examples),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--src-lang", required=True, choices=list(LANG_MAP.keys()))
    parser.add_argument("--tgt-lang", required=True, choices=list(LANG_MAP.keys()))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-json", required=True)

    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    dtype = resolve_torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    examples = load_flores_examples(args.src_lang, args.tgt_lang, args.limit)

    hook = build_quant_hook(model, args.model_key, args.mode, args.bits)

    try:
        metrics = evaluate(model, tokenizer, examples, args.src_lang, args.tgt_lang)
    finally:
        if hook:
            hook.remove()

    result = {
        "model_key": args.model_key,
        "mode": args.mode,
        "bits": args.bits,
        "dtype": args.dtype,
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "num_examples": metrics["num_examples"],
        "chrf": metrics.get("chrf"),
        "loss": metrics.get("loss"),
        "ppl": metrics.get("ppl"),
        "predictions": metrics["predictions"],
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "model_key": args.model_key,
                "mode": args.mode,
                "bits": args.bits,
                "src_lang": args.src_lang,
                "tgt_lang": args.tgt_lang,
                "num_examples": metrics["num_examples"],
                "chrf": metrics.get("chrf"),
                "loss": metrics.get("loss"),
                "ppl": metrics.get("ppl"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()