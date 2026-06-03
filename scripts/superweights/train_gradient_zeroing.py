from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


OLMO1B_SUPERWEIGHTS = [
    {"layer": 1, "row": 1764, "col": 1710},
    {"layer": 1, "row": 1764, "col": 8041},
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


def get_superweight_gradients(model, superweights) -> dict[str, float | None]:
    gradients = {}

    for sw in superweights:
        layer = sw["layer"]
        row = sw["row"]
        col = sw["col"]

        weight = get_olmo_down_proj_weight(model, layer)
        key = f"layer{layer}_row{row}_col{col}"

        if weight.grad is None:
            gradients[key] = None
        else:
            gradients[key] = weight.grad[row, col].float().item()

    return gradients


def zero_superweight_gradients(model, superweights) -> None:
    for sw in superweights:
        layer = sw["layer"]
        row = sw["row"]
        col = sw["col"]

        weight = get_olmo_down_proj_weight(model, layer)

        if weight.grad is None:
            raise RuntimeError(
                f"No gradient found for layer {layer} down_proj.weight"
            )

        weight.grad[row, col] = 0.0


def print_superweight_gradients(model, superweights, title: str) -> None:
    print(f"\n{title}")

    gradients = get_superweight_gradients(model, superweights)

    for key, value in gradients.items():
        print(f"{key}: grad={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini training with optional superweight gradient zeroing."
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "grad_zero"],
        default="grad_zero",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON results.",
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

    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
    )

    texts = [
        "The capital of France is Paris.",
        "Large language models predict the next token.",
        "Quantization reduces memory usage in neural networks.",
        "Superweights are unusually important parameters.",
        "Gradient zeroing prevents selected weights from updating.",
    ]

    initial_values = get_superweight_values(model, OLMO1B_SUPERWEIGHTS)
    step_logs = []

    print("\nInitial superweight values")
    for key, value in initial_values.items():
        print(f"{key}: {value}")

    for step in range(args.max_steps):
        text = texts[step % len(texts)]

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            **inputs,
            labels=inputs["input_ids"],
        )

        loss = outputs.loss
        loss.backward()

        gradients_before = get_superweight_gradients(
            model,
            OLMO1B_SUPERWEIGHTS,
        )

        print(f"\n{'=' * 80}")
        print(f"Step {step + 1}/{args.max_steps}")
        print(f"Loss: {loss.item():.4f}")

        print_superweight_gradients(
            model,
            OLMO1B_SUPERWEIGHTS,
            title="Before optional gradient zeroing",
        )

        gradients_after = None

        if args.mode == "grad_zero":
            zero_superweight_gradients(
                model,
                OLMO1B_SUPERWEIGHTS,
            )

            gradients_after = get_superweight_gradients(
                model,
                OLMO1B_SUPERWEIGHTS,
            )

            print_superweight_gradients(
                model,
                OLMO1B_SUPERWEIGHTS,
                title="After gradient zeroing",
            )
        else:
            gradients_after = gradients_before
            print("Baseline mode: gradients are not modified.")

        optimizer.step()

        step_logs.append(
            {
                "step": step + 1,
                "text": text,
                "loss": float(loss.item()),
                "gradients_before": gradients_before,
                "gradients_after": gradients_after,
            }
        )

    final_values = get_superweight_values(model, OLMO1B_SUPERWEIGHTS)

    print("\nFinal superweight values")
    for key, value in final_values.items():
        print(f"{key}: {value}")

    deltas = {}
    print("\nSuperweight deltas")
    for key in initial_values:
        delta = final_values[key] - initial_values[key]
        deltas[key] = delta
        print(f"{key}: delta={delta}")

    results = {
        "model": model_name,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "optimizer": "SGD",
        "superweights": OLMO1B_SUPERWEIGHTS,
        "initial_values": initial_values,
        "final_values": final_values,
        "deltas": deltas,
        "steps": step_logs,
    }

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON results to {output_path}")

    print("\nMini training finished.")


if __name__ == "__main__":
    main()