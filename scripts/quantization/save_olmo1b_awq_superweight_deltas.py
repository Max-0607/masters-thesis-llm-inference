import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from superweight_awq_patch import save_awq_deltas
from quantization.awq.awq.quantize.pre_quant import apply_awq
from quantization.awq.awq.quantize.quantizer import pseudo_quantize_model_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--awq-path", required=True)
    parser.add_argument("--fp16-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False

    _ = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    q_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
    }

    print("Loading AWQ results...")
    awq_results = torch.load(args.awq_path, map_location="cpu")

    print("Applying AWQ...")
    apply_awq(model, awq_results)

    print("Pseudo-quantizing weights...")
    pseudo_quantize_model_weight(
        model,
        w_bit=args.w_bit,
        q_config=q_config,
    )

    print("Computing deltas...")
    deltas = save_awq_deltas(
        model,
        args.fp16_json,
        args.output_json,
    )

    print(deltas)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()