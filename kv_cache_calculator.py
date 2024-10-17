
"""
KV Cache Calculator Script
--------------------------
This script calculates the size of the KV cache for a transformer model based on
model configuration details like context length, hidden size, attention heads, key-value heads, 
number of layers, and precision.

The KV cache is used to store key and value representations for each token during the forward pass 
in transformer models. Efficient management of KV cache memory is crucial for handling large models, 
especially when working with long sequences or multiple GPUs.

"""

def calculate_kv_cache_size(context_length, hidden_size, num_attention_heads, num_key_value_heads, num_hidden_layers, precision_bytes):
    """
    Calculate the size of the KV cache based on model configuration parameters.

    Parameters:
        - context_length: The maximum number of tokens the model can handle.
        - hidden_size: The hidden dimension size per layer.
        - num_attention_heads: The total number of attention heads.
        - num_key_value_heads: The number of key-value heads (for parallelism).
        - num_hidden_layers: The number of hidden layers in the model.
        - precision_bytes: Bytes per parameter (0.5 for 4-bit quantization).

    Returns:
        The KV cache size in GB.
    """
    kv_cache_params = 2 * context_length * hidden_size * (num_key_value_heads / num_attention_heads) * num_hidden_layers
    kv_cache_size_bytes = kv_cache_params * precision_bytes
    kv_cache_size_gb = kv_cache_size_bytes / (1024**3)  # Convert bytes to GB
    return kv_cache_size_gb


if __name__ == "__main__":
    # Example with sample values from Qwen config with AWQ 4-bit quantization https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ/blob/main/config.json
    context_length = 32768  # Max position embeddings
    hidden_size = 8192  # Hidden size (dim per layer)
    num_attention_heads = 64  # Total number of attention heads
    num_key_value_heads = 8  # Number of key-value heads
    num_hidden_layers = 80  # Number of hidden layers
    precision_bytes = 0.5  # AWQ 4-bit precision (0.5 bytes per parameter)

    # Calculate the KV cache size
    kv_cache_size = calculate_kv_cache_size(context_length, hidden_size, num_attention_heads, num_key_value_heads, num_hidden_layers, precision_bytes)
    
    print(f"KV Cache Size: {kv_cache_size:.2f} GB")
