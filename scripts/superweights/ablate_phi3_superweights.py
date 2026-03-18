import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

SUPER_WEIGHTS = [
    (2, 525, 808),
    (2, 1693, 808),
    (2, 1113, 808),
    (4, 525, 2723),
    (4, 1113, 2723),
    (4, 1693, 2723),
]

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("Ablating superweights...")

with torch.no_grad():
    for layer, row, col in SUPER_WEIGHTS:

        weight = model.model.layers[layer].mlp.down_proj.weight

        print(f"Ablating layer {layer} weight[{row},{col}]")

        weight[row, col] = 0

print("Done.")

save_dir = "outputs/phi3_superweights_ablated_model"
model.save_pretrained(save_dir)
print(f"Saved ablated model to {save_dir}")
