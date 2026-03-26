MODEL_CONFIGS = {
    "olmo-1b": {
        "hf_name": "allenai/OLMo-1B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
    },
    "phi-3": {
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
    },
}