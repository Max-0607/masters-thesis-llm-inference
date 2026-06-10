from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from src.hooks import get_nested_attr


MODEL_SPECS = {
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(1, 2070, 7310)],
    },
    "llama-7b": {
        "hf_name": "huggyllama/llama-7b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(2, 3968, 7003)],
    },
    "olmo1b": {
        "hf_name": "allenai/OLMo-1B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(1, 1764, 1710), (1, 1764, 8041)],
    },
    "olmo7b": {
        "hf_name": "allenai/OLMo-7B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(1, 269, 7467), (2, 269, 8275), (7, 269, 453), (24, 269, 2300)],
    },
    "phi3-mini": {
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (2, 525, 808), (2, 1693, 808), (2, 1113, 808),
            (4, 525, 2723), (4, 1113, 2723), (4, 1693, 2723),
        ],
    },
}


def to_sw_dicts(tuples):
    return [{"layer": l, "row": r, "col": c} for l, r, c in tuples]


def load_wikitext_texts(split: str, limit: int) -> list[str]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = []
    for item in dataset:
        text = item["text"].strip()
        if len(text) > 50 and not text.startswith("="):
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def get_input_device(model):
    return next(model.parameters()).device


def get_layers(model, layer_path: str):
    return get_nested_attr(model, layer_path)


def get_down_proj_weight(model, layer: int, layer_path: str, down_proj_path: str):
    layers = get_layers(model, layer_path)
    module = get_nested_attr(layers[layer], down_proj_path)
    return module.weight


def get_superweight_values(model, superweights, layer_path, down_proj_path):
    values = {}
    for sw in superweights:
        weight = get_down_proj_weight(model, sw["layer"], layer_path, down_proj_path)
        key = f"layer{sw['layer']}_row{sw['row']}_col{sw['col']}"
        values[key] = weight.data[sw["row"], sw["col"]].detach().float().cpu().item()
    return values


def get_superweight_gradients(model, superweights, layer_path, down_proj_path):
    gradients = {}
    for sw in superweights:
        weight = get_down_proj_weight(model, sw["layer"], layer_path, down_proj_path)
        key = f"layer{sw['layer']}_row{sw['row']}_col{sw['col']}"
        gradients[key] = None if weight.grad is None else weight.grad[sw["row"], sw["col"]].detach().float().cpu().item()
    return gradients


def zero_superweight_gradients(model, superweights, layer_path, down_proj_path):
    for sw in superweights:
        weight = get_down_proj_weight(model, sw["layer"], layer_path, down_proj_path)
        if weight.grad is None:
            raise RuntimeError(f"No gradient found for layer {sw['layer']} down_proj.weight")
        weight.grad[sw["row"], sw["col"]] = 0.0


def save_superweight_values(model, superweights, layer_path, down_proj_path):
    saved = {}
    for sw in superweights:
        weight = get_down_proj_weight(model, sw["layer"], layer_path, down_proj_path)
        saved[(sw["layer"], sw["row"], sw["col"])] = weight.data[sw["row"], sw["col"]].detach().clone()
    return saved


def restore_superweight_values(model, saved_values, layer_path, down_proj_path):
    with torch.no_grad():
        for (layer, row, col), value in saved_values.items():
            weight = get_down_proj_weight(model, layer, layer_path, down_proj_path)
            weight.data[row, col] = value.to(weight.device)


def maybe_dropout_superweights(model, superweights, dropout_prob, layer_path, down_proj_path):
    if torch.rand(1).item() >= dropout_prob:
        return False
    with torch.no_grad():
        for sw in superweights:
            weight = get_down_proj_weight(model, sw["layer"], layer_path, down_proj_path)
            weight.data[sw["row"], sw["col"]] = 0.0
    return True


def freeze_all_except_superweight_layers(model, superweights, layer_path):
    layers_to_train = sorted({sw["layer"] for sw in superweights})
    layers = get_layers(model, layer_path)

    for param in model.parameters():
        param.requires_grad = False

    for layer_idx in layers_to_train:
        for param in layers[layer_idx].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable layers: {layers_to_train}")
    print(f"Trainable params: {trainable:,} / {total:,}")
    print(f"Trainable ratio: {trainable / total:.6f}")


def compute_mean_loss(model, tokenizer, texts, max_length):
    model.eval()
    losses = []
    input_device = get_input_device(model)

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(input_device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(float(outputs.loss.item()))

    model.train()
    return sum(losses) / len(losses)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-key", choices=list(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--mode", choices=["baseline", "grad_zero", "sw_dropout"], required=True)

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sw-dropout-prob", type=float, default=0.25)

    parser.add_argument("--train-only-superweight-layers", action="store_true")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--save-model-dir", type=str, default=None)

    return parser.parse_args()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    spec = MODEL_SPECS[args.model_key]

    model_name = args.model_name or spec["hf_name"]
    layer_path = spec["layer_path"]
    down_proj_path = spec["down_proj_path"]
    superweights = to_sw_dicts(spec["superweights"])

    print(f"Model key: {args.model_key}")
    print(f"Loading model: {model_name}")
    print(f"Mode: {args.mode}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"SW dropout prob: {args.sw_dropout_prob}")
    print(f"Train only SW layers: {args.train_only_superweight_layers}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts = load_wikitext_texts("train", args.train_samples)
    eval_texts = load_wikitext_texts("validation", args.eval_samples)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model.train()
    input_device = get_input_device(model)
    print(f"Input device: {input_device}")

    if args.train_only_superweight_layers:
        freeze_all_except_superweight_layers(model, superweights, layer_path)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    initial_values = get_superweight_values(model, superweights, layer_path, down_proj_path)

    print("\nInitial superweight values")
    for key, value in initial_values.items():
        print(f"{key}: {value}")

    loss_before_training = compute_mean_loss(
        model=model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
    )
    print(f"\nEval loss before training: {loss_before_training:.4f}")

    step_logs = []

    for step in range(args.max_steps):
        text = train_texts[step % len(train_texts)]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(input_device)

        optimizer.zero_grad(set_to_none=True)

        saved_sw_values = None
        dropout_applied = False

        if args.mode == "sw_dropout":
            saved_sw_values = save_superweight_values(model, superweights, layer_path, down_proj_path)
            dropout_applied = maybe_dropout_superweights(
                model,
                superweights,
                args.sw_dropout_prob,
                layer_path,
                down_proj_path,
            )

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        gradients_before = get_superweight_gradients(model, superweights, layer_path, down_proj_path)

        if args.mode == "grad_zero":
            zero_superweight_gradients(model, superweights, layer_path, down_proj_path)

        gradients_after = get_superweight_gradients(model, superweights, layer_path, down_proj_path)

        optimizer.step()

        if args.mode == "sw_dropout" and saved_sw_values is not None:
            restore_superweight_values(model, saved_sw_values, layer_path, down_proj_path)

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"[{args.mode}] step {step + 1}/{args.max_steps} "
                f"loss={loss.item():.4f} sw_dropout={dropout_applied}"
            )

        step_logs.append(
            {
                "step": step + 1,
                "text": text,
                "loss": float(loss.item()),
                "sw_dropout_applied": bool(dropout_applied),
                "gradients_before": gradients_before,
                "gradients_after": gradients_after,
            }
        )

    final_values = get_superweight_values(model, superweights, layer_path, down_proj_path)
    deltas = {key: final_values[key] - initial_values[key] for key in initial_values}

    loss_after_training = compute_mean_loss(
        model=model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
    )
    print(f"\nEval loss after training: {loss_after_training:.4f}")

    results = {
        "model": model_name,
        "model_key": args.model_key,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "optimizer": "SGD",
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "max_length": args.max_length,
        "sw_dropout_prob": args.sw_dropout_prob,
        "train_only_superweight_layers": args.train_only_superweight_layers,
        "sw_dropout_count": sum(1 for x in step_logs if x["sw_dropout_applied"]),
        "layer_path": layer_path,
        "down_proj_path": down_proj_path,
        "superweights": superweights,
        "initial_values": initial_values,
        "final_values": final_values,
        "deltas": deltas,
        "loss_before_training": loss_before_training,
        "loss_after_training": loss_after_training,
        "train_logs": step_logs,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON results to {output_path}")

    if args.save_model_dir is not None:
        save_dir = Path(args.save_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved trained model to {save_dir}")

    print("\nMini training finished.")


if __name__ == "__main__":
    main()