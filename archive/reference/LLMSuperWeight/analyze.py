import matplotlib.pyplot as plt
import torch
from clize import run
from outliers.functional.utils import add_global_plot_styles
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_module_hidden_states(model, tokenizer, test_text, layer_path, module_name, input_or_output="output", plot_fname=None, spike_threshold=100):
    if input_or_output not in ["input", "output"]:
        raise ValueError("input_or_output should be 'input' or 'output', instead of", input_or_output)
    
    all_activations = {}

    def get_activations(layer_index):
        def hook(model, inputs, outputs):
            hidden_states = inputs if input_or_output == "input" else outputs
            all_activations.setdefault(layer_index, {})[f"{module_name}_{input_or_output}_hidden_states"] = hidden_states
        return hook   

    all_hooks = []

    def get_layers(model, layer_path):
        attributes = layer_path.split('.')
        layers = model
        for attr in attributes:
            layers = getattr(layers, attr)
        return layers

    attributes = module_name.split('.') if module_name != "layer" else []
    layers = get_layers(model, layer_path)

    for layer_index, layer in enumerate(layers):
        current_attr = layer
        valid = True
        for attr in attributes:
            if hasattr(current_attr, attr):
                current_attr = getattr(current_attr, attr)
            else:
                valid = False
                break
        
        if valid:
            hook = current_attr.register_forward_hook(get_activations(layer_index))
            all_hooks.append(hook)

    inputs = tokenizer(test_text, return_tensors='pt').to(model.device)
    model.eval()
    with torch.no_grad():
        model(**inputs)

    for hook in all_hooks:
        hook.remove()

    top1_values_all_layers = []
    top1_indexes_all_layers = []
    for layer_index, outputs in all_activations.items():
        values = outputs[f'{module_name}_{input_or_output}_hidden_states']
        tensor = values[0] if isinstance(values, tuple) else values
        tensor = tensor.detach().cpu()
        tensor_abs = tensor.view(-1).abs().float()

        max_value, max_index = torch.max(tensor_abs, 0)
        max_index = torch.unravel_index(max_index, tensor.shape)
        top1_values_all_layers.append(tensor[max_index])
        top1_indexes_all_layers.append(max_index)

    return top1_values_all_layers, top1_indexes_all_layers

def plot_down_proj_input_output(pretrained="allenai/OLMo-7B-0724-hf", module_name="mlp.down_proj"):
    model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

    test_text = "Apple Inc. is a worldwide tech company."
    layer_path = "model.layers"

    for name in ("input", "output"):
        magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)

        # Report any spikes
        spikes_input = [i for i, value in enumerate(magnitude) if abs(value) > 50]
        print(f"Activation spikes for {module_name} {name}:")
        for i in spikes_input:
            spike_index = index[i]
            print(f" - layer {i}, value {magnitude[i]}, index {tuple(i.item() for i in spike_index)}")

        # Plot input activations
        plt.figure(figsize=(5,3.5))
        add_global_plot_styles()
        plt.plot(range(len(magnitude)), magnitude, color='blue', marker='o', markersize=5)
        plt.xlabel('Layer Number')
        plt.ylabel('Max Activation Value')
        plt.title(f"OLMo-7B Max down_proj {name}")
        plt.yticks(rotation=90, va='center')
        plt.savefig(f"outputs/figures/{name}_down_proj.pdf", bbox_inches='tight')
        print(f"Plot saved to 'outputs/figures/{name}_down_proj.pdf'")

        # Print output magnitudes
        print(f"largest_activations_down_proj_{name}={list(map(float, magnitude))}")

def record_SO(model, pretrained):
    '''Record SO values for original models'''
    SUPER_WEIGHTS_MAP = {
        "Mistral-7B-v0.1": [(1, 2070, 7310)],
        "llama-7B": [(2, 3968, 7003)],
        "llama-13B": [(2, 2231, 2278), (2, 2231, 6939)],
        "llama-30B": [(3, 5633, 12817), (3, 5633, 17439), (10, 5633, 14386)],
        "Meta-Llama-3-8B": [(1, 788, 2427), (1, 1384, 2427), (1, 4062, 2427)],
        "OLMo-1B-0724-hf": [(1, 1764, 1710), (2, 1764, 8041)],
        "OLMo-7B-0724-hf": [(1, 269, 7467), (2, 269, 8275), (7, 269, 453), (24, 269, 2300)],
        "Phi-3-mini-4k-instruct": [(2, 525, 808), (2, 1693, 808), (2, 1113, 808), (4, 525, 2723),  (4, 1113, 2723), (4, 1693, 2723)],
    }
    
    def _record_SO(SO_map, layer, row, col):
        if pretrained in [
            "tiiuae/falcon-7b",
        ]:
            SO_map[(layer, row, col)] = model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
        else:
            SO_map[(layer, row, col)] = model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()

    SO_values = {}
    for model_name, coordinates in SUPER_WEIGHTS_MAP.items():
        if model_name in pretrained:
            for layer, row, col in coordinates:
                _record_SO(SO_values, layer, row, col)
            break
    return SO_values

def scale_SO(model, pretrained, SO_values, scaling_factor):
    if pretrained in [
        "huggyllama/llama-30B", 
        "huggyllama/llama-13B", 
        "huggyllama/llama-7B", 
        "mistralai/Mistral-7B-v0.1",
        "meta/Meta-Llama-3-8B",
        "allenai/OLMo-1B-0724-hf",
        "allenai/OLMo-7B-0724-hf",
        "microsoft/Phi-3-mini-4k-instruct"
        ]:
        for (layer, row, col), value in SO_values.items():  
            old_value = model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()
            new_value = value * scaling_factor
            model.model.layers[layer].mlp.down_proj.weight.data[row, col] = new_value
            print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")

def remove_outliers(model, pretrained, percentage_threshold):

    num_selected_elements = []
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue

        weight = param.data
        num_top_elements = int(weight.numel() * percentage_threshold)
        # print("# Total params:", weight.numel(), "# top params", num_top_elements)
        if num_top_elements < 1: # too few elements to apply on
            continue 
        threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]
        mask = weight.abs() >= threshold
        true_indices = mask.nonzero(as_tuple=False)
        num_selected_elements.append(len(true_indices))

        weight[mask] = 0.
        param = torch.nn.Parameter(weight)   
    


def plot_max_activation_ablation(pretrained="allenai/OLMo-7B-0724-hf"):
    model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

    
    test_text = "Apple Inc. is a worldwide tech company."
    layer_path = "model.layers"
    module_name = "layer"
    name = "output"
    # original
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Print output magnitudes
    print(f"original={list(map(float, magnitude))}")
    
    # remove SO
    SO_values = record_SO(model, pretrained)
    scale_SO(model, pretrained, SO_values, 0)
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Report any spikes
    spikes_input = [i for i, value in enumerate(magnitude) if abs(value) > 50]
    print(f"Activation spikes for {module_name} {name}:")
    for i in spikes_input:
        spike_index = index[i]
        print(f" - layer {i}, value {magnitude[i]}, index {tuple(i.item() for i in spike_index)}")
    # Print output magnitudes
    print(f"super_weight_removed={list(map(float, magnitude))}")

    # remove outliers
    percentage = 5e-7
    remove_outliers(model, pretrained, percentage)
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Print output magnitudes
    print(f"all_outliers_removed={list(map(float, magnitude))}")

    # restore SO
    scale_SO(model, pretrained, SO_values, 1)
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Print output magnitudes
    print(f"all_other_outliers_removed={list(map(float, magnitude))}")

def plot_token_probs(pretrained="mistralai/Mistral-7B-v0.1"):

    model_name = pretrained
    MODEL_ID = model_name.split('/')[1]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    )

    model_map = {
        "Mistral-7B-v0.1": [(1, 2070, 7310)],
        "llama-7B": [(2, 3968, 7003)],
        "llama-13B": [(2, 2231, 2278), (2, 2231, 6939)],
        "llama-3v0B": [(3, 5633, 12817), (3, 5633, 17439), (10, 5633, 14386)],
        "Meta-Llama-3-8B": [(1, 788, 2427), (1, 1384, 2427), (1, 4062, 2427)],
        "OLMo-1B-0724-hf": [(1, 1764, 1710), (2, 1764, 8041)],
        "OLMo-7B-0724-hf": [(1, 269, 7467), (2, 269, 8275), (7, 269, 453), (24, 269, 2300)],
        "gemma-7b": [(0, 1995, 21041)], # not sufficient
        "Phi-3-mini-4k-instruct": [(2, 525, 808), (2, 1693, 808), (2, 1113, 808), (4, 525, 2723),  (4, 1113, 2723), (4, 1693, 2723)],
        # "tiiuae/falcon-7b": [(3, 2002, 10708), (4, 2002, 5921)]
    }
    sw_map = {}

    def remove_SO(model):
        sw_map[MODEL_ID] = []
        for (layerno, y, x) in model_map[MODEL_ID]:
            weight = model.model.layers[layerno].mlp.down_proj.weight.data
            sw_map[MODEL_ID].append(float(weight[y, x]))
            weight[y, x] = 0.
            model.model.layers[layerno].mlp.down_proj.weight = torch.nn.Parameter(weight)

    def restore_SO(model):
        assert sw_map.get(MODEL_ID, None), "Run remove_SO before running restore_SO"
        for value, (layerno, y, x) in zip(sw_map[MODEL_ID], model_map[MODEL_ID]):
            weight = model.model.layers[layerno].mlp.down_proj.weight.data
            weight[y, x] = value
            model.model.layers[layerno].mlp.down_proj.weight = torch.nn.Parameter(weight)

    def print_SO(model):
        for weight, (layerno, y, x) in zip(sw_map[MODEL_ID], model_map[MODEL_ID]):
            weight = model.model.layers[layerno].mlp.down_proj.weight.data
            print(weight[y, x])


    def get_next_token_probs(model, tokenizer, input_text):
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids)
            
            # Get the logits for the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
        return next_token_probs


    from datasets import load_dataset
    import json
    from tqdm import tqdm

    N_SAMPLES = 500
    dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")

    all_difference = []

    all_probs_SO_removed = []
    all_probs_Original = []


    # Original model
    for text in tqdm(dataset[:N_SAMPLES]["text"]):
        prompt = ' '.join(text.split(' ')[:-1])
        target = text.split(' ')[-1]
        next_token_probs = get_next_token_probs(model, tokenizer, prompt)
        all_probs_Original.append(next_token_probs)

    avg_probs_Original = (sum(all_probs_Original)  / len(all_probs_Original))[0] 
    avg_probs_Original = avg_probs_Original.to('cpu')
    sorted_probs_Original, sorted_indices_Original = torch.sort(avg_probs_Original, descending=True)
    top_n = 100
    top_n_probs_Original = sorted_probs_Original[:top_n].tolist()
    top_n_indices_Original = sorted_indices_Original[:top_n]
    top_tokens = [tokenizer.decode(i) for i in top_n_indices_Original]

    # Remove super weight
    remove_SO(model)
    print(sw_map[MODEL_ID])

    for text in tqdm(dataset[:N_SAMPLES]["text"]):
        prompt = ' '.join(text.split(' ')[:-1])
        target = text.split(' ')[-1]
        next_token_probs = get_next_token_probs(model, tokenizer, prompt)
        all_probs_SO_removed.append(next_token_probs)

    # Average probabilities acorss all samples
    avg_probs_SO_removed = (sum(all_probs_SO_removed) / len(all_probs_SO_removed))[0]
    avg_probs_SO_removed = avg_probs_SO_removed.to('cpu')
    selected_token_probs_SO_removed = []
    for i in top_n_indices_Original:
        selected_token_probs_SO_removed.append(avg_probs_SO_removed[i].item())

    print("Top n tokens:")
    print(top_tokens)
    print("Original")
    print(top_n_probs_Original)
    print("SW removed")
    print(selected_token_probs_SO_removed)
    



if __name__ == '__main__':
    run([plot_down_proj_input_output, plot_max_activation_ablation, plot_token_probs])
