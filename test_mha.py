import torch
import torch.nn as nn

print(f"Using PyTorch version: {torch.__version__}")

try:
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
    print("Successfully initialized MultiheadAttention with batch_first=True")
except Exception as e:
    print(f"Failed to initialize MultiheadAttention: {e}")
