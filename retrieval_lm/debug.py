import torch
import torch.nn as nn

# Define the embedding layer
vocab_size = 128256
hidden_size = 2048
embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=0).cuda()

# Create a sample input_ids tensor
input_ids = torch.randint(0, vocab_size, (1, 879)).cuda()

# Check the input_ids tensor
print(f"input_ids shape: {input_ids.shape}")
print(f"input_ids values: {input_ids}")
print(f"input_ids device: {input_ids.device}")
print(f"embedding layer device: {embed_tokens.weight.device}")

# Check for out-of-bounds indices
if torch.any(input_ids >= vocab_size):
    raise ValueError(f"input_ids contain out-of-bounds indices. vocab_size: {vocab_size}, input_ids: {input_ids}")
else:
    print("No out-of-bounds indices in input_ids")

# Check for NaN or Inf values in input_ids
if torch.any(torch.isnan(input_ids)) or torch.any(torch.isinf(input_ids)):
    raise ValueError(f"input_ids contain NaN or Inf values. input_ids: {input_ids}")

# Check for NaN or Inf values in embedding weights
if torch.any(torch.isnan(embed_tokens.weight)) or torch.any(torch.isinf(embed_tokens.weight)):
    raise ValueError(f"Embedding weights contain NaN or Inf values. embedding weights: {embed_tokens.weight}")

# Perform the embedding operation
try:
    inputs_embeds = embed_tokens(input_ids)
    print(f"inputs_embeds shape: {inputs_embeds.shape}")
except Exception as e:
    print(f"Error during embedding operation: {e}")
    
