import torch



def pack_4bit_to_int8(quantized_4bit_weights):
    # Ensure the quantized weights are uint8
    quantized_4bit_weights = quantized_4bit_weights.to(torch.uint8)
    
    # Reshape to ensure even number of elements per row
    if quantized_4bit_weights.size(1) % 2 != 0:
        quantized_4bit_weights = torch.cat((quantized_4bit_weights, torch.zeros((quantized_4bit_weights.size(0), 1), dtype=torch.uint8)), dim=1)
    
    # Packing two 4-bit values into one int8
    packed_weights = (quantized_4bit_weights[:, ::2] << 4) | (quantized_4bit_weights[:, 1::2] & 0xF)
    
    return packed_weights



def unpack_int8_to_4bit(packed_weights):
    # Unpack two 4-bit values from one int8
    high_bits = (packed_weights >> 4) & 0xF
    low_bits = packed_weights & 0xF
    
    # Interleave the high and low bits
    unpacked_weights = torch.stack((high_bits, low_bits), dim=2).flatten(1)
    
    return unpacked_weights

def dequantize_4bit_to_fp16(quantized, maxima, minima, nbits, blocksize):
    shape = quantized.shape
    
    if quantized.numel() % (blocksize * blocksize) == 0:
        quantized = quantized.reshape(-1, blocksize * blocksize)
        scale = (2 ** nbits - 1) / (maxima - minima)
        
        dequantized = (quantized / scale) + minima
        dequantized = dequantized.reshape(shape)
    else:
        scale = (2 ** nbits - 1) / (maxima - minima)
        dequantized = (quantized / scale) + minima

    return dequantized


def round_to_nearest_pole(x, poles):
    # Compute the absolute differences between each element of x and all poles
    differences = torch.abs(x.unsqueeze(-1) - poles)
    
    # Find the index of the minimum difference along the poles dimension
    nearest_indices = torch.argmin(differences, dim=-1)
    
    # Return the corresponding nearest pole values for each element in x
    nearest_values = poles[nearest_indices]
    
    return nearest_values


def quantize_blockwise(weight, nbits, blocksize, clip_method, clip_threshold, scale_shift=False, use_normal_float=False):
    # weight should be param.data
    shape = weight.shape
    dtype = weight.dtype
    _num_outliers = 0
    if weight.numel() % (blocksize) == 0:
        weight = weight.reshape(-1, blocksize)
        # block wise
        if clip_method != "no":

            if clip_method == "block_percentage":
                num_top_elements = max(int(weight.size(1) * clip_threshold + 1), 1) # per block
                threshold = torch.topk(weight.abs(), num_top_elements, dim=1).values[:, -1].unsqueeze(-1)

            elif clip_method == "tensor_percentage":
                num_top_elements = max(int(weight.numel() * clip_threshold), 1)
                threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]

            elif clip_method == "zscore":
                means = weight.abs().mean(dim=1, keepdim=True)
                stds = weight.abs().std(dim=1, keepdim=True)
                threshold = clip_threshold * stds + means

            elif clip_method == "iqr":
                q1 = weight.abs().float().quantile(0.25, dim=1, keepdim=True)
                q3 = weight.abs().float().quantile(0.75, dim=1, keepdim=True)
                threshold = q3 + clip_threshold * (q3 - q1) # 1.5 * IQR
                threshold = threshold.to(weight.dtype)
            else:
                raise ValueError(f"Unknown clip method: {clip_method}")

            _num_outliers = int(torch.sum(weight.abs() > threshold ))
            weight = torch.clamp(weight, -threshold, threshold)
        minima, _ = weight.min(dim=1, keepdims=True)
        maxima, _ = weight.max(dim=1, keepdims=True)

    else:
        if clip_method != "no":
            if clip_method == "tensor_percentage" or clip_method == "block_percentage":
                num_top_elements = max(int(weight.numel() * clip_threshold), 1)
                threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]
            elif clip_method == "zscore":
                mean = weight.abs().mean()
                std = weight.abs().std()
                threshold = clip_threshold * std + mean 
            elif clip_method == "iqr":
                q1 = weight.abs().float().quantile(0.25)
                q3 = weight.abs().float().quantile(0.75)
                threshold = q3 + clip_threshold * (q3 - q1) # 1.5 * IQR
            # count clipped outliers
            _num_outliers = int(torch.sum(weight.abs() > threshold))
            weight = torch.clamp(weight, -threshold, threshold)
        minima, maxima = weight.min(), weight.max()


    if use_normal_float: # NF4 or NF3
        NF4_LEVELS = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
        
        NF3_LEVELS = [-1, -0.5350227355957031, -
        0.2469314038753510, 0, 0.1833375245332718, 0.3819939494132996, 0.6229856610298157, 1]
        if nbits == 4:
            quantization_levels = NF4_LEVELS
        elif nbits == 3:
            quantization_levels = NF3_LEVELS
        else:
            raise ValueError("Normal Float Quantization only suuports 4 and 3 bits now.")
        quantization_levels = torch.tensor(quantization_levels).to(weight.device)
        scale = 2 / (maxima - minima) # scale to [0, 2]
        weight.sub_(minima).mul_(scale).sub_(1.0) # shift to [-1, 1]
        weight.copy_(round_to_nearest_pole(weight, quantization_levels))
        weight.add_(1).div_(scale).add_(minima)


    else: # INT
        if scale_shift:
            # mapping weights to [-0.4999, 15.4999] and then round to nearest integers
            scale = (2 ** nbits - 0.01) / (maxima - minima)
            weight.sub_(minima).mul_(scale).sub_(0.49)
            weight.round_()
            weight.add_(0.49).div_(scale).add_(minima)
        else:
            # mapping weights to [0, 15] and then round to nearest integers
            scale = (2 ** nbits - 1) / (maxima - minima)
            weight.sub_(minima).mul_(scale)
            weight.round_()
            weight.div_(scale).add_(minima)

    dequantized = weight.reshape(shape).to(dtype)
    return dequantized, _num_outliers
        