"""NCF: kết hợp GMF + MLP."""

import torch
import torch.nn as nn


class NCFModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 32,
        mlp_dims=(64, 32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        # GMF
        self.gmf_user_emb = nn.Embedding(num_users, gmf_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_dim)

        # MLP
        self.mlp_emb_dim = mlp_dims[0] // 2
        self.mlp_user_emb = nn.Embedding(num_users, self.mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, self.mlp_emb_dim)

        layers, in_dim = [], mlp_dims[0]
        for out_dim in mlp_dims[1:]:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.final = nn.Linear(gmf_dim + mlp_dims[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user, item):
        gmf_out = self.gmf_user_emb(user) * self.gmf_item_emb(item)
        mlp_in = torch.cat([self.mlp_user_emb(user), self.mlp_item_emb(item)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        out = self.final(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1)
        return out
