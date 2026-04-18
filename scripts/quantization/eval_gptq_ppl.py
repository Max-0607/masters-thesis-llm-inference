#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


MODEL_IDS = {
    "llama-7b": "huggyllama/llama-7b",
    "olmo-1b": "allenai/OLMo-1B-0724-hf",
    "olmo-7b": "allenai/OLMo-7B-0724-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
}


@dataclass
class ParsedRun:
    perplexities: Dict[str, float]
    raw_stdout: str


def parse_ppls(stdout: str) -> Dict[str, float]:
    """
    Parse old GPTQ llama.py output of the form:

    wikitext2
    Evaluating ...
    8.8469
    ptb
    Evaluating ...
    12.34
    c4
    Evaluating ...
    7.1908
    """
    lines = [line.strip() for line in stdout.splitlines()]
    datasets = {"wikitext2", "ptb", "ptb-new", "c4", "c4-new"}
    ppls: Dict[str, float] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        if line in datasets:
            dataset = line
            # search forward for first float-looking line
            j = i + 1
            while j < len(lines):
                candidate = lines[j]
                if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", candidate):
                    ppls[dataset] = float(candidate)
                    break
                # if another dataset starts first, stop
                if candidate in datasets:
                    break
                j += 1
            i = j
        i += 1

    return ppls


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> ParsedRun:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    captured_lines = []

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_lines.append(line)

    return_code = process.wait()
    stdout = "".join(captured_lines)

    if return_code != 0:
        raise RuntimeError(f"GPTQ run failed with exit code {return_code}")

    return ParsedRun(
        perplexities=parse_ppls(stdout),
        raw_stdout=stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, help="e.g. llama-7b")
    parser.add_argument("--model-id", default=None, help="HF model id override")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-dataset", default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--new-eval", action="store_true")
    parser.add_argument("--save-pt", default=None, help="Path to save quantized checkpoint")
    parser.add_argument("--log-file", default=None, help="Optional raw log output file")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--gptq-repo",
        default="quantization/gptq/gptq_repo",
        help="Path to old GPTQ repo containing llama.py",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    gptq_repo = (project_root / args.gptq_repo).resolve()
    output_json = (project_root / args.output_json).resolve() if not Path(args.output_json).is_absolute() else Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    model_id = args.model_id or MODEL_IDS.get(args.model_key, args.model_key)

    cmd = [
        sys.executable,
        "llama.py",
        model_id,
        args.calibration_dataset,
        "--wbits",
        str(args.bits),
        "--groupsize",
        str(args.groupsize),
        "--nsamples",
        str(args.nsamples),
        "--seed",
        str(args.seed),
    ]

    if args.true_sequential:
        cmd.append("--true-sequential")
    if args.act_order:
        cmd.append("--act-order")
    if args.sym:
        cmd.append("--sym")
    if args.new_eval:
        cmd.append("--new-eval")
    if args.save_pt:
        save_pt = (project_root / args.save_pt).resolve() if not Path(args.save_pt).is_absolute() else Path(args.save_pt)
        save_pt.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--save", str(save_pt)])
    else:
        save_pt = None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    parsed = run_command(cmd, cwd=gptq_repo, env=env)

    result = {
        "model_key": args.model_key,
        "model_id": model_id,
        "method": "gptq",
        "bits": args.bits,
        "groupsize": args.groupsize,
        "nsamples": args.nsamples,
        "seed": args.seed,
        "calibration_dataset": args.calibration_dataset,
        "true_sequential": args.true_sequential,
        "act_order": args.act_order,
        "sym": args.sym,
        "new_eval": args.new_eval,
        "checkpoint_path": str(save_pt) if save_pt else None,
        "perplexity": parsed.perplexities,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.log_file:
        log_file = (project_root / args.log_file).resolve() if not Path(args.log_file).is_absolute() else Path(args.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(parsed.raw_stdout, encoding="utf-8")

    print("\nSaved JSON to:", output_json)
    if args.log_file:
        print("Saved log to:", log_file)


if __name__ == "__main__":
    main()