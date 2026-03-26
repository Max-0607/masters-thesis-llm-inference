import os
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map={"": "cuda:0"},
    low_cpu_mem_usage=True,
)

print("loaded ok")
print(type(model).__name__)
