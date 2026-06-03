from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch


OLMO1B_SUPERWEIGHTS = [
    {"layer": 1, "row": 1764, "col": 1710},
    {"layer": 1, "row": 1764, "col": 8041},
]


TRAIN_TEXTS = [
    "The capital of France is Paris.",
    "Large language models predict the next token.",
    "Quantization reduces memory usage in neural networks.",
    "Superweights are unusually important parameters.",
    "Gradient zeroing prevents selected weights from updating.",
]


EVAL_TEXTS = [
    "The capital of Germany is Berlin.",
    "Neural networks learn patterns from data.",
    "Language models generate text token by token.",
    "Model compression can reduce memory requirements.",
    "Important parameters may strongly affect model behavior.",
]


def get_olmo_down_proj_weight(model, layer: int) -> torch.nn.Parameter:
    return model.model.layers[layer].mlp.down_proj.weight


def get_superweight_values(model, superweights) -> dict[str, float]:
    values = {}

    for sw in superweights:
        layer = sw["layer"]
        row = sw["row"]
        col = sw["col"]

        weight = get_olmo_down_proj_weight(model, layer)
        key = f"layer{layer}_row{row}_col{col}"
        values[key] = weight.data[row, col].float().item()

    return values


def zero_superweight_gradients(model, superweights) -> None:
    for sw in superweights:
        weight = get_olmo_down_proj_weight(model, sw["layer"])

        if weight.grad is None:
            raise RuntimeError(
                f"No gradient found for layer {sw['layer']} down_proj.weight"
            )

        weight.grad[sw["row"], sw["col"]] = 0.0


def ablate_superweights(model, superweights) -> None:
    with torch.no_grad():
        for sw in superweights:
            weight = get_olmo_down_proj_weight(model, sw["layer"])
            weight.data[sw["row"], sw["col"]] = 0.0


def compute_mean_loss(model, tokenizer, texts: list[str], device: str) -> float:
    model.eval()

    losses = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(float(outputs.loss.item()))

    model.train()
    return sum(losses) / len(losses)


def mini_train(model, tokenizer, mode: str, max_steps: int, learning_rate: float, device: str):
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    losses = []

    for step in range(max_steps):
        text = TRAIN_TEXTS[step % len(TRAIN_TEXTS)]
        inputs = tokenizer(text, return_tensors="pt").to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        if mode == "grad_zero":
            zero_superweight_gradients(model, OLMO1B_SUPERWEIGHTS)

        optimizer.step()

        losses.append(float(loss.item()))

        if (step + 1) % 10 == 0 or step == 0:
            print(f"[{mode}] step {step + 1}/{max_steps} loss={loss.item():.4f}")

    return losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness after superweight ablation."
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "grad_zero"],
        required=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()

    model_name = "allenai/OLMo-1B-0724-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    initial_superweight_values = get_superweight_values(
        model,
        OLMO1B_SUPERWEIGHTS,
    )

    loss_before_training = compute_mean_loss(
        model,
        tokenizer,
        EVAL_TEXTS,
        device,
    )

    print(f"Eval loss before training: {loss_before_training:.4f}")

    train_losses = mini_train(
        model=model,
        tokenizer=tokenizer,
        mode=args.mode,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=device,
    )

    final_superweight_values = get_superweight_values(
        model,
        OLMO1B_SUPERWEIGHTS,
    )

    loss_after_training_before_ablation = compute_mean_loss(
        model,
        tokenizer,
        EVAL_TEXTS,
        device,
    )

    print(
        "Eval loss after training before ablation: "
        f"{loss_after_training_before_ablation:.4f}"
    )

    ablated_model = deepcopy(model)
    ablate_superweights(
        ablated_model,
        OLMO1B_SUPERWEIGHTS,
    )

    loss_after_ablation = compute_mean_loss(
        ablated_model,
        tokenizer,
        EVAL_TEXTS,
        device,
    )

    print(f"Eval loss after ablation: {loss_after_ablation:.4f}")

    ablation_delta = loss_after_ablation - loss_after_training_before_ablation
    print(f"Ablation loss delta: {ablation_delta:.4f}")

    superweight_deltas = {}
    for key in initial_superweight_values:
        superweight_deltas[key] = (
            final_superweight_values[key] - initial_superweight_values[key]
        )

    results = {
        "model": model_name,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "optimizer": "SGD",
        "superweights": OLMO1B_SUPERWEIGHTS,
        "initial_superweight_values": initial_superweight_values,
        "final_superweight_values": final_superweight_values,
        "superweight_deltas": superweight_deltas,
        "loss_before_training": loss_before_training,
        "loss_after_training_before_ablation": loss_after_training_before_ablation,
        "loss_after_ablation": loss_after_ablation,
        "ablation_loss_delta": ablation_delta,
        "train_losses": train_losses,
        "train_texts": TRAIN_TEXTS,
        "eval_texts": EVAL_TEXTS,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()