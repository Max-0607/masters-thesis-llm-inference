import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from superweight_awq_patch import save_fp16_superweights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--output-json", default="results/deployment_olmo1b/fp16_superweights.json")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    values = save_fp16_superweights(model, args.output_json)
    print(values)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()