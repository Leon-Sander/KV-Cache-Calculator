
# KV Cache Calculator

This repository contains a script to calculate the Key-Value (KV) cache size for transformer models. 
The KV cache is an important component in transformer models, especially during inference. 
It stores key and value representations for tokens processed by the model, which helps in efficient memory management and faster inference.

## What is the KV Cache?

In transformer models, each token is represented as a key and a value in the attention mechanism. 
The **KV cache** stores these key-value pairs during the forward pass, allowing the model to reuse these cached representations 
for subsequent token predictions instead of recalculating them. This is especially useful for tasks like autoregressive generation.

The size of the KV cache depends on various factors, including:
- **Context length** (number of tokens)
- **Hidden size** (dimension of each layer)
- **Number of attention heads**
- **Number of key-value heads** (for parallelism)
- **Number of hidden layers** in the model
- **Precision** (e.g., 4-bit, 16-bit) used for storing the cache

## How It Works

The size of the KV cache can be calculated using the following formula:

```
KV Cache Size = 2 * context_length * hidden_size * (key_value_heads / attention_heads) * number_of_layers
```

The `2` factor accounts for both the key and value representations.

## Python Script

The provided Python script calculates the KV cache size for a transformer model. 
You can modify the model parameters to match your specific model's configuration and run the script to calculate the KV cache size.

### Example usage

```bash
python kv_cache_calculator.py
```

By default, the script is set to calculate the KV cache size for the [Qwen model](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ) with 4-bit quantization. 
You can modify the parameters in the script to fit your own model's configuration.
