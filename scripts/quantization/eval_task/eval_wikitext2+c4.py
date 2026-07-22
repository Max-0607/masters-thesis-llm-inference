import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.models import MODEL_CONFIGS
from configs.superweights import SUPERWEIGHTS
from src.hooks import get_nested_attr


def resolve_torch_dtype(name: str):
    name = name.lower()

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def make_json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]
    return obj


def get_superweight_restore_indices(
    row: int,
    col: int,
    shape,
    neighborhood: str = "scalar",
):
    n_rows, n_cols = shape
    indices = set()

    def add(r, c):
        if 0 <= r < n_rows and 0 <= c < n_cols:
            indices.add((r, c))

    add(row, col)

    if neighborhood in ["row", "cross"]:
        add(row, col - 1)
        add(row, col + 1)

    if neighborhood in ["column", "cross"]:
        add(row - 1, col)
        add(row + 1, col)

    return sorted(indices)


@torch.no_grad()
def quantize_activation(x: torch.Tensor, n_bits: int):
    if not torch.is_floating_point(x):
        return x

    orig_dtype = x.dtype
    x_float = x.float()

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    max_abs = x_float.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.clamp(
        torch.round(x_float / scale),
        qmin,
        qmax,
    )

    return (q * scale).to(orig_dtype)


def add_activation_hooks(model, n_bits: int):
    handles = []

    def hook(module, inputs):
        if not inputs:
            return inputs

        x = inputs[0]

        if not torch.is_tensor(x):
            return inputs

        return (
            quantize_activation(x, n_bits),
        ) + tuple(inputs[1:])

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_pre_hook(hook)
            )

    return handles


@torch.no_grad()
def clip_weight_tensor_zscore(
    weight: torch.Tensor,
    clip_z: Optional[float],
):
    if clip_z is None or clip_z <= 0:
        return weight

    orig_dtype = weight.dtype
    weight_float = weight.float()

    mean = weight_float.mean()
    std = weight_float.std(unbiased=False)

    if std.item() == 0:
        return weight

    lower = mean - clip_z * std
    upper = mean + clip_z * std

    return torch.clamp(
        weight_float,
        lower,
        upper,
    ).to(orig_dtype)


@torch.no_grad()
def quantize_weight_tensor(
    weight: torch.Tensor,
    n_bits: int,
):
    """
    Asymmetric min-max round-to-nearest quantization.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    orig_dtype = weight.dtype
    weight_float = weight.float()

    qmin = 0
    qmax = (2 ** n_bits) - 1

    weight_min = weight_float.min()
    weight_max = weight_float.max()

    if (weight_max - weight_min).item() == 0:
        return weight_float.to(orig_dtype)

    delta = (weight_max - weight_min) / qmax

    quantized = torch.clamp(
        torch.round(
            (weight_float - weight_min) / delta
        ),
        qmin,
        qmax,
    )

    return (
        quantized * delta + weight_min
    ).to(orig_dtype)


@torch.no_grad()
def quantize_weight_tensor_blockwise_2d(
    weight: torch.Tensor,
    n_bits: int,
    block_rows: int = 128,
    block_cols: int = 128,
):
    """
    Blockwise asymmetric min-max quantization.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(
            "block_rows and block_cols must be positive"
        )

    if weight.ndim != 2:
        return quantize_weight_tensor(
            weight,
            n_bits,
        )

    orig_dtype = weight.dtype
    weight_float = weight.float()
    output = torch.empty_like(weight_float)

    qmin = 0
    qmax = (2 ** n_bits) - 1

    n_rows, n_cols = weight_float.shape

    for row_start in range(0, n_rows, block_rows):
        row_end = min(
            row_start + block_rows,
            n_rows,
        )

        for col_start in range(0, n_cols, block_cols):
            col_end = min(
                col_start + block_cols,
                n_cols,
            )

            block = weight_float[
                row_start:row_end,
                col_start:col_end,
            ]

            block_min = block.min()
            block_max = block.max()

            if (block_max - block_min).item() == 0:
                output[
                    row_start:row_end,
                    col_start:col_end,
                ] = block
                continue

            delta = (block_max - block_min) / qmax

            quantized = torch.clamp(
                torch.round(
                    (block - block_min) / delta
                ),
                qmin,
                qmax,
            )

            output[
                row_start:row_end,
                col_start:col_end,
            ] = quantized * delta + block_min

    return output.to(orig_dtype)


@torch.no_grad()
def quantize_parameter(
    param: torch.Tensor,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    weight = clip_weight_tensor_zscore(
        param,
        clip_z,
    )

    if quant_granularity == "tensor":
        return quantize_weight_tensor(
            weight,
            n_bits,
        )

    if quant_granularity == "block2d":
        return quantize_weight_tensor_blockwise_2d(
            weight,
            n_bits=n_bits,
            block_rows=block_rows,
            block_cols=block_cols,
        )

    raise ValueError(
        f"Unsupported quant_granularity: "
        f"{quant_granularity}"
    )


@torch.no_grad()
def apply_weight_quantization(
    model,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    quantized_modules = 0
    skipped_modules = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if "lm_head" in module_name:
            skipped_modules += 1
            continue

        if "embed" in module_name:
            skipped_modules += 1
            continue

        quantized_weight = quantize_parameter(
            module.weight.data,
            n_bits=n_bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=clip_z,
        )

        module.weight.data.copy_(quantized_weight)
        quantized_modules += 1

    print(
        "Weight quantization done: "
        f"quantized Linear modules={quantized_modules}, "
        f"skipped Linear modules={skipped_modules}",
        flush=True,
    )

    return model


@torch.no_grad()
def collect_superweights(
    model,
    model_key: str,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
):
    if model_key not in SUPERWEIGHTS:
        raise ValueError(
            "No superweights registered for "
            f"model_key={model_key}"
        )

    model_config = MODEL_CONFIGS[model_key]

    layers = get_nested_attr(
        model,
        model_config["layer_path"],
    )

    down_proj_path = model_config["down_proj_path"]
    protected = []

    for entry in SUPERWEIGHTS[model_key]:
        layer_idx = int(entry["layer"])
        row = int(entry["row"])
        col = int(entry["col"])

        module = get_nested_attr(
            layers[layer_idx],
            down_proj_path,
        )

        weight = module.weight.data

        restore_indices = (
            get_superweight_restore_indices(
                row=row,
                col=col,
                shape=weight.shape,
                neighborhood=restore_neighborhood,
            )
        )

        for restore_row, restore_col in restore_indices:
            is_center = (
                restore_row == row
                and restore_col == col
            )

            value = (
                weight[
                    restore_row,
                    restore_col,
                ]
                .detach()
                .clone()
            )

            if is_center:
                value = value * sw_scale

            protected.append(
                {
                    "layer": layer_idx,
                    "center_row": row,
                    "center_col": col,
                    "row": int(restore_row),
                    "col": int(restore_col),
                    "value": value,
                    "is_center": bool(is_center),
                }
            )

    return protected


@torch.no_grad()
def restore_superweights(
    model,
    model_key: str,
    protected,
):
    model_config = MODEL_CONFIGS[model_key]

    layers = get_nested_attr(
        model,
        model_config["layer_path"],
    )

    down_proj_path = model_config["down_proj_path"]

    for item in protected:
        module = get_nested_attr(
            layers[item["layer"]],
            down_proj_path,
        )

        module.weight.data[
            item["row"],
            item["col"],
        ] = item["value"]

    return model


def prepare_model(
    model,
    model_key: str,
    mode: str,
    bits: int,
    activation_bits: Optional[int],
    sw_scale: float,
    restore_neighborhood: str = "scalar",
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    activation_handles = []
    protected = []

    if mode == "fp16":
        return model, protected, activation_handles

    if mode == "naive":
        apply_weight_quantization(
            model=model,
            n_bits=bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=None,
        )

        if (
            activation_bits is not None
            and activation_bits > 0
        ):
            activation_handles = add_activation_hooks(
                model,
                activation_bits,
            )

        return model, protected, activation_handles

    if mode == "super":
        protected = collect_superweights(
            model=model,
            model_key=model_key,
            sw_scale=sw_scale,
            restore_neighborhood=restore_neighborhood,
        )

        apply_weight_quantization(
            model=model,
            n_bits=bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=clip_z,
        )

        restore_superweights(
            model=model,
            model_key=model_key,
            protected=protected,
        )

        if (
            activation_bits is not None
            and activation_bits > 0
        ):
            activation_handles = add_activation_hooks(
                model,
                activation_bits,
            )

        return model, protected, activation_handles

    raise ValueError(f"Unsupported mode: {mode}")


def load_text_pool(
    dataset: str,
    split: str,
    pool_size: int,
):
    """
    Load a fixed pool of non-empty evaluation texts.
    """
    if pool_size <= 0:
        raise ValueError(
            "sampling_pool_size must be positive"
        )

    if dataset == "wikitext2":
        dataset_object = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=split,
        )

    elif dataset == "c4":
        dataset_object = load_dataset(
            "allenai/c4",
            "en",
            split=split,
            streaming=True,
        )

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    pool = []

    for original_idx, example in enumerate(dataset_object):
        text = example.get("text", "")

        if not text.strip():
            continue

        pool.append(
            {
                "id": int(original_idx),
                "text": text,
            }
        )

        if len(pool) >= pool_size:
            break

    if not pool:
        raise RuntimeError(
            f"No non-empty texts found for {dataset}."
        )

    print(
        f"Loaded pool with {len(pool)} non-empty texts",
        flush=True,
    )

    return pool


def select_texts(
    pool,
    limit: int,
    eval_seed: int,
    reference_json: Optional[str] = None,
):
    """
    Select texts reproducibly or reuse the example IDs
    stored in an FP16 reference JSON.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    if limit > len(pool):
        raise ValueError(
            f"limit={limit} exceeds available "
            f"pool size={len(pool)}"
        )

    pool_by_id = {
        int(item["id"]): item["text"]
        for item in pool
    }

    if reference_json is None:
        rng = random.Random(eval_seed)

        available_ids = [
            int(item["id"])
            for item in pool
        ]

        selected_ids = rng.sample(
            available_ids,
            limit,
        )

    else:
        reference_path = Path(reference_json)

        if not reference_path.exists():
            raise FileNotFoundError(
                "Reference JSON does not exist: "
                f"{reference_path}"
            )

        with open(
            reference_path,
            "r",
            encoding="utf-8",
        ) as file:
            reference_result = json.load(file)

        selected_ids = reference_result.get(
            "selected_example_ids"
        )

        if selected_ids is None:
            raise ValueError(
                "Reference JSON has no "
                f"selected_example_ids: {reference_path}"
            )

        selected_ids = [
            int(example_id)
            for example_id in selected_ids
        ]

        if len(selected_ids) != limit:
            raise ValueError(
                "Reference JSON contains "
                f"{len(selected_ids)} IDs, "
                f"but limit={limit}"
            )

        reference_dataset = reference_result.get(
            "dataset"
        )

        if reference_dataset != reference_result.get(
            "dataset",
            reference_dataset,
        ):
            raise ValueError(
                "Reference dataset does not match."
            )

        reference_split = reference_result.get("split")

        if reference_split is not None:
            print(
                "Using reference selection from "
                f"dataset={reference_dataset}, "
                f"split={reference_split}",
                flush=True,
            )

        missing_ids = [
            example_id
            for example_id in selected_ids
            if example_id not in pool_by_id
        ]

        if missing_ids:
            raise ValueError(
                "Reference IDs are not available in "
                f"the loaded pool: {missing_ids[:10]}"
            )

    texts = [
        pool_by_id[example_id]
        for example_id in selected_ids
    ]

    return texts, selected_ids


@torch.no_grad()
def evaluate_ppl(
    model,
    tokenizer,
    texts,
    max_length: int,
):
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    evaluated_texts = 0

    model.eval()

    for index, text in enumerate(texts):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoded.input_ids.to(device)

        if input_ids.shape[1] < 2:
            continue

        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
            return_dict=True,
        )

        num_tokens = input_ids.shape[1] - 1
        loss = outputs.loss.float().item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens
        evaluated_texts += 1

        if (index + 1) % 25 == 0:
            current_ppl = math.exp(
                total_loss / total_tokens
            )

            print(
                f"{index + 1}/{len(texts)} | "
                f"current_ppl={current_ppl:.4f}",
                flush=True,
            )

    if total_tokens == 0:
        raise RuntimeError(
            "No valid tokens evaluated."
        )

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
        "num_texts": evaluated_texts,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )

    parser.add_argument(
        "--mode",
        default="fp16",
        choices=["fp16", "naive", "super"],
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--activation-bits",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sw-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--restore-neighborhood",
        type=str,
        default="scalar",
        choices=[
            "scalar",
            "row",
            "column",
            "cross",
        ],
    )

    parser.add_argument(
        "--quant-granularity",
        type=str,
        default="tensor",
        choices=["tensor", "block2d"],
    )

    parser.add_argument(
        "--block-rows",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--block-cols",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--clip-z",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["wikitext2", "c4"],
    )

    parser.add_argument(
        "--split",
        default="validation",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed used to select evaluation texts.",
    )

    parser.add_argument(
        "--reference-json",
        type=str,
        default=None,
        help=(
            "Reuse selected_example_ids from an "
            "FP16 reference JSON."
        ),
    )

    parser.add_argument(
        "--sampling-pool-size",
        type=int,
        default=2000,
        help=(
            "Number of non-empty texts from which "
            "the evaluation sample is selected."
        ),
    )

    parser.add_argument(
        "--dtype",
        default="float16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    model_config = MODEL_CONFIGS[args.model_key]
    model_id = model_config["hf_name"]

    torch_dtype = resolve_torch_dtype(
        args.dtype
    )

    print(
        f"Loading tokenizer: {model_id}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Loading model: {model_id}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    model.eval()

    print(
        f"Preparing model mode={args.mode}, "
        f"bits={args.bits}, "
        f"activation_bits={args.activation_bits}, "
        f"sw_scale={args.sw_scale}, "
        f"restore_neighborhood="
        f"{args.restore_neighborhood}, "
        f"quant_granularity="
        f"{args.quant_granularity}, "
        f"block_rows={args.block_rows}, "
        f"block_cols={args.block_cols}, "
        f"clip_z={args.clip_z}",
        flush=True,
    )

    model, protected, activation_handles = prepare_model(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
        activation_bits=args.activation_bits,
        sw_scale=args.sw_scale,
        restore_neighborhood=args.restore_neighborhood,
        quant_granularity=args.quant_granularity,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        clip_z=args.clip_z,
    )

    print(
        f"Protected values: {len(protected)}",
        flush=True,
    )

    print(
        f"Activation hooks: "
        f"{len(activation_handles)}",
        flush=True,
    )

    print(
        f"Loading dataset={args.dataset}, "
        f"split={args.split}, "
        f"pool_size={args.sampling_pool_size}, "
        f"limit={args.limit}, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    pool = load_text_pool(
        dataset=args.dataset,
        split=args.split,
        pool_size=args.sampling_pool_size,
    )

    texts, selected_example_ids = select_texts(
        pool=pool,
        limit=args.limit,
        eval_seed=args.eval_seed,
        reference_json=args.reference_json,
    )

    print(
        f"Selected {len(texts)} texts from "
        f"a pool of {len(pool)} texts",
        flush=True,
    )

    print(
        "First selected example IDs: "
        f"{selected_example_ids[:10]}",
        flush=True,
    )

    metrics = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
    )

    protected_summary = [
        {
            "layer": int(item["layer"]),
            "center_row": int(
                item["center_row"]
            ),
            "center_col": int(
                item["center_col"]
            ),
            "row": int(item["row"]),
            "col": int(item["col"]),
            "is_center": bool(
                item["is_center"]
            ),
        }
        for item in protected
    ]

    result = {
        "benchmark": "perplexity",
        "dataset": args.dataset,
        "model_key": args.model_key,
        "model_id": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "activation_bits": args.activation_bits,
        "sw_scale": args.sw_scale,
        "restore_neighborhood":
            args.restore_neighborhood,
        "quant_granularity":
            args.quant_granularity,
        "block_rows": args.block_rows,
        "block_cols": args.block_cols,
        "clip_z": args.clip_z,
        "num_protected_values": len(protected),
        "protected_values": protected_summary,
        "num_activation_hooks":
            len(activation_handles),
        "dtype": args.dtype,
        "split": args.split,
        "limit": args.limit,
        "max_length": args.max_length,
        "eval_seed": args.eval_seed,
        "reference_json": args.reference_json,
        "sampling_pool_size":
            args.sampling_pool_size,
        "selected_example_ids":
            selected_example_ids,
        **metrics,
    }

    result = make_json_safe(result)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(
        f"Saved result to: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()