from __future__ import annotations

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

    for sw in superweights:
        layer = sw["layer"]
        row = sw["row"]
        col = sw["col"]

        weight = get_olmo_down_proj_weight(model, layer)

        grad_value = None
        if weight.grad is not None:
            grad_value = weight.grad[row, col].item()

        print(f"layer={layer}, row={row}, col={col}, grad={grad_value}")


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "allenai/OLMo-1B-0724-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_steps = 10
    learning_rate = 1e-5

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    texts = [
        "The capital of France is Paris.",
        "Large language models predict the next token.",
        "Quantization reduces memory usage in neural networks.",
        "Superweights are unusually important parameters.",
        "Gradient zeroing prevents selected weights from updating.",
    ]

    initial_values = get_superweight_values(model, OLMO1B_SUPERWEIGHTS)

    print("\nInitial superweight values")
    for key, value in initial_values.items():
        print(f"{key}: {value}")

    for step in range(max_steps):
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

        print(f"\n{'=' * 80}")
        print(f"Step {step + 1}/{max_steps}")
        print(f"Loss: {loss.item():.4f}")

        print_superweight_gradients(
            model,
            OLMO1B_SUPERWEIGHTS,
            title="Before gradient zeroing",
        )

        zero_superweight_gradients(
            model,
            OLMO1B_SUPERWEIGHTS,
        )

        print_superweight_gradients(
            model,
            OLMO1B_SUPERWEIGHTS,
            title="After gradient zeroing",
        )

        optimizer.step()

    final_values = get_superweight_values(model, OLMO1B_SUPERWEIGHTS)

    print("\nFinal superweight values")
    for key, value in final_values.items():
        print(f"{key}: {value}")

    print("\nSuperweight deltas")
    for key in initial_values:
        delta = final_values[key] - initial_values[key]
        print(f"{key}: delta={delta}")

    print("\nMini training finished.")


if __name__ == "__main__":
    main()