"""DeepCoNN: 2 nhánh CNN song song + FM."""

import torch
import torch.nn as nn


class DeepCoNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        out_channels: int = 64,
        kernel_size: int = 3,
        latent_dim: int = 32,
        dropout: float = 0.3,
        pretrained_emb: torch.Tensor = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        self.user_conv = nn.Conv1d(embed_dim, out_channels, kernel_size)
        self.item_conv = nn.Conv1d(embed_dim, out_channels, kernel_size)
        self.user_fc = nn.Linear(out_channels, latent_dim)
        self.item_fc = nn.Linear(out_channels, latent_dim)
        self.dropout = nn.Dropout(dropout)

        self.fm_linear = nn.Linear(2 * latent_dim, 1)

    def _branch(self, x_ids, conv, fc):
        emb = self.embedding(x_ids).transpose(1, 2)
        h = torch.relu(conv(emb))
        h = torch.max(h, dim=2).values
        h = self.dropout(h)
        return torch.relu(fc(h))

    def forward(self, user_ids, item_ids):
        u = self._branch(user_ids, self.user_conv, self.user_fc)
        i = self._branch(item_ids, self.item_conv, self.item_fc)

        linear_out = self.fm_linear(torch.cat([u, i], dim=-1)).squeeze(-1)

        v = torch.stack([u, i], dim=1)
        sum_v = v.sum(dim=1)
        sum_v_sq = (v**2).sum(dim=1)
        fm_factor = 0.5 * (sum_v**2 - sum_v_sq).sum(dim=-1)
        return linear_out + fm_factor
