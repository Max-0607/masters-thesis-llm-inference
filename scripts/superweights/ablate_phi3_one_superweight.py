import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
SAVE_DIR = "outputs/phi3_one_superweight_ablated_model"

# first Phi-3 superweight from the paper
LAYER, ROW, COL = 2, 525, 808

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Ablating layer {LAYER} weight[{ROW},{COL}]")
with torch.no_grad():
    weight = model.model.layers[LAYER].mlp.down_proj.weight
    weight[ROW, COL] = 0

print("Saving...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved to {SAVE_DIR}")
