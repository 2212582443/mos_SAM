import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.att = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
        )

    def forward(self, x):
        return x
