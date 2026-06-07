import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm
from datasets import load_dataset


@dataclass
class MCQExample:
    example_id: str
    context: str
    choices: List[str]
    label: int


def ensure_parent_dir(path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_choice_loglikelihood(
    model,
    tokenizer,
    prompt: str,
    choice: str,
    device: torch.device,
) -> float:
    """
    Returns the conditional log-likelihood of `choice` given `prompt`.
    We mask the prompt tokens so only the choice contributes to the score.
    """
    full_text = prompt + choice

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

    input_ids = full_ids["input_ids"].to(device)
    attention_mask = full_ids["attention_mask"].to(device)

    labels = input_ids.clone()
    prompt_len = prompt_ids["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    loss = outputs.loss
    num_target_tokens = (labels != -100).sum().item()

    # Convert average token loss into total log-likelihood
    total_loglik = -loss.item() * num_target_tokens
    return total_loglik


def evaluate_mcq_dataset(
    model,
    tokenizer,
    examples: List[MCQExample],
    limit: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    if limit is not None:
        examples = examples[:limit]

    num_correct = 0
    predictions = []

    iterator = tqdm(examples, desc="Evaluating", total=len(examples))

    for ex in iterator:
        scores = [
            score_choice_loglikelihood(
                model=model,
                tokenizer=tokenizer,
                prompt=ex.context,
                choice=choice,
                device=device,
            )
            for choice in ex.choices
        ]

        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        correct = int(pred == ex.label)
        num_correct += correct

        row = {
            "id": ex.example_id,
            "label": ex.label,
            "prediction": pred,
            "correct": correct,
            "scores": scores,
            "context": ex.context if verbose else None,
            "choices": ex.choices if verbose else None,
        }
        predictions.append(row)

        iterator.set_postfix(acc=f"{num_correct / len(predictions):.4f}")

    accuracy = num_correct / max(len(predictions), 1)

    return {
        "num_examples": len(predictions),
        "num_correct": num_correct,
        "accuracy": accuracy,
        "predictions": predictions,
    }


def save_results(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)