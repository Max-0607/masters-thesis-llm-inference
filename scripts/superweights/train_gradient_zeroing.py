from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset


SUPERWEIGHTS = {
    "olmo1b": [
        {"layer": 1, "row": 1764, "col": 1710},
        {"layer": 1, "row": 1764, "col": 8041},
    ],
    "olmo7b": [
        {"layer": 1, "row": 269, "col": 7467},
        {"layer": 2, "row": 269, "col": 8275},
        {"layer": 7, "row": 269, "col": 453},
        {"layer": 24, "row": 269, "col": 2300},
    ],
}


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


def get_olmo_down_proj_weight(model, layer: int):
    return model.model.layers[layer].mlp.down_proj.weight


def get_superweight_values(model, superweights) -> dict[str, float]:
    values = {}
    for sw in superweights:
        weight = get_olmo_down_proj_weight(model, sw["layer"])
        key = f"layer{sw['layer']}_row{sw['row']}_col{sw['col']}"
        values[key] = weight.data[sw["row"], sw["col"]].detach().float().cpu().item()
    return values


def get_superweight_gradients(model, superweights) -> dict[str, float | None]:
    gradients = {}
    for sw in superweights:
        weight = get_olmo_down_proj_weight(model, sw["layer"])
        key = f"layer{sw['layer']}_row{sw['row']}_col{sw['col']}"
        if weight.grad is None:
            gradients[key] = None
        else:
            gradients[key] = weight.grad[sw["row"], sw["col"]].detach().float().cpu().item()
    return gradients


def zero_superweight_gradients(model, superweights) -> None:
    for sw in superweights:
        weight = get_olmo_down_proj_weight(model, sw["layer"])
        if weight.grad is None:
            raise RuntimeError(f"No gradient found for layer {sw['layer']} down_proj.weight")
        weight.grad[sw["row"], sw["col"]] = 0.0


def save_superweight_values(model, superweights):
    saved = {}
    for sw in superweights:
        layer, row, col = sw["layer"], sw["row"], sw["col"]
        weight = get_olmo_down_proj_weight(model, layer)
        saved[(layer, row, col)] = weight.data[row, col].detach().clone()
    return saved


def restore_superweight_values(model, saved_values) -> None:
    with torch.no_grad():
        for (layer, row, col), value in saved_values.items():
            weight = get_olmo_down_proj_weight(model, layer)
            weight.data[row, col] = value.to(weight.device)


def maybe_dropout_superweights(model, superweights, dropout_prob: float) -> bool:
    if torch.rand(1).item() >= dropout_prob:
        return False

    with torch.no_grad():
        for sw in superweights:
            weight = get_olmo_down_proj_weight(model, sw["layer"])
            weight.data[sw["row"], sw["col"]] = 0.0
    return True


def freeze_all_except_superweight_layers(model, superweights) -> None:
    layers_to_train = sorted({sw["layer"] for sw in superweights})

    for param in model.parameters():
        param.requires_grad = False

    for layer_idx in layers_to_train:
        for param in model.model.layers[layer_idx].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable layers: {layers_to_train}")
    print(f"Trainable params: {trainable:,} / {total:,}")
    print(f"Trainable ratio: {trainable / total:.6f}")


def compute_mean_loss(model, tokenizer, texts, max_length: int) -> float:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-type", choices=["olmo1b", "olmo7b"], required=True)
    parser.add_argument("--model-name", type=str, required=True)
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


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    superweights = SUPERWEIGHTS[args.model_type]

    print(f"Model type: {args.model_type}")
    print(f"Loading model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train only SW layers: {args.train_only_superweight_layers}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts = load_wikitext_texts("train", args.train_samples)
    eval_texts = load_wikitext_texts("validation", args.eval_samples)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model.train()
    input_device = get_input_device(model)
    print(f"Input device: {input_device}")

    if args.train_only_superweight_layers:
        freeze_all_except_superweight_layers(model, superweights)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    initial_values = get_superweight_values(model, superweights)

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
            saved_sw_values = save_superweight_values(model, superweights)
            dropout_applied = maybe_dropout_superweights(
                model,
                superweights,
                args.sw_dropout_prob,
            )

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        gradients_before = get_superweight_gradients(model, superweights)

        if args.mode == "grad_zero":
            zero_superweight_gradients(model, superweights)

        gradients_after = get_superweight_gradients(model, superweights)

        optimizer.step()

        if args.mode == "sw_dropout" and saved_sw_values is not None:
            restore_superweight_values(model, saved_sw_values)

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

    final_values = get_superweight_values(model, superweights)
    deltas = {key: final_values[key] - initial_values[key] for key in initial_values}

    loss_after_training = compute_mean_loss(
        model=model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
    )
    print(f"\nEval loss after training: {loss_after_training:.4f}")

    results = {
        "model": args.model_name,
        "model_type": args.model_type,
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