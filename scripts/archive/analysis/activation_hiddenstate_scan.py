import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_SETS = {
    "reasoning": [
        "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost?"
    ],
    "math": [
        "27 * 14 = ?"
    ],
    "causal": [
        "Why does dropping a glass on the floor make it shatter?"
    ],
    "knowledge": [
        "What is the capital of France?"
    ],
    "coding": [
        "Write a Python function that returns the factorial of a number."
    ],
}


def analyze_hidden_states(model, tok, texts, max_len=32):
    device = next(model.parameters()).device
    layer_maxima = None

    with torch.no_grad():
        for t in texts:
            inp = tok(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            inp = {k: v.to(device) for k, v in inp.items()}

            out = model(
                **inp,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )

            # hidden_states: tuple length = n_layers + 1
            # hidden_states[0] = embeddings, hidden_states[1:] = layer outputs
            hs = out.hidden_states[1:]

            current = []
            for h in hs:
                # [batch, seq, hidden] -> max abs over batch/seq per layer
                val = h.detach().float().abs().max().item()
                current.append(val)

            current = torch.tensor(current)

            if layer_maxima is None:
                layer_maxima = current
            else:
                layer_maxima = torch.maximum(layer_maxima, current)

    spike_layer = int(torch.argmax(layer_maxima).item())

    return {
        "layer_maxima": layer_maxima.tolist(),
        "spike_layer": spike_layer,
        "spike_value": float(layer_maxima[spike_layer].item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--category", type=str, required=True, choices=list(PROMPT_SETS.keys()))
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    result = analyze_hidden_states(
        model,
        tok,
        PROMPT_SETS[args.category],
        max_len=args.max_len,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
