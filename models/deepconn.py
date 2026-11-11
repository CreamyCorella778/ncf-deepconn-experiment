"""DeepCoNN: 2 nhánh CNN song song + FM."""

import torch
import torch.nn as nn


class DeepCoNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_users: int,
        n_items: int,
        embed_dim: int = 100,
        out_channels: int = 64,
        kernel_size: int = 3,
        latent_dim: int = 32,
        dropout: float = 0.3,
        pretrained_emb: torch.Tensor = None,
    ):
        super().__init__()
        if pretrained_emb is not None:
            if pretrained_emb.shape != (vocab_size, embed_dim):
                raise ValueError(
                    "pretrained_emb must have shape "
                    f"({vocab_size}, {embed_dim}), got {tuple(pretrained_emb.shape)}."
                )
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_emb.float(), freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        self.user_id_emb = nn.Embedding(n_users, latent_dim)
        self.item_id_emb = nn.Embedding(n_items, latent_dim)
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

    def forward(self, user_idx, item_idx, user_reviews, item_reviews):
        u = self._branch(user_reviews, self.user_conv, self.user_fc)
        i = self._branch(item_reviews, self.item_conv, self.item_fc)
        u = u + self.user_id_emb(user_idx)
        i = i + self.item_id_emb(item_idx)
        return self._combine(u, i)

    def score_candidates(
        self, user_idx, item_indices, user_reviews, item_reviews
    ):
        """Score candidate items, reusing each user's text and ID representations."""
        batch_size, candidate_count, max_len = item_reviews.shape
        u = self._branch(user_reviews, self.user_conv, self.user_fc)
        u = u + self.user_id_emb(user_idx)
        flat_items = item_reviews.reshape(batch_size * candidate_count, max_len)
        i = self._branch(flat_items, self.item_conv, self.item_fc)
        i = i.reshape(batch_size, candidate_count, -1)
        i = i + self.item_id_emb(item_indices)
        u = u.unsqueeze(1).expand_as(i)
        return self._combine(u, i)

    def _combine(self, u, i):
        linear_out = self.fm_linear(torch.cat([u, i], dim=-1)).squeeze(-1)
        return linear_out + torch.sum(u * i, dim=-1)
