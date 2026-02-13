"""
Transformer-based models for pMSSM regression.

This module provides transformer architectures optimized for tabular pMSSM data.
"""

import torch
import torch.nn as nn


class PMSSMTransformer(nn.Module):
    """
    Improved transformer with positional encoding to preserve feature order.

    Pros: Maintains feature identity, learns interactions
    Cons: Still might be overkill for small feature set
    """
    def __init__(
        self,
        n_params=19,
        d_model=64,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.0,  # Changed default to 0.0 for small datasets
        use_prenorm=True,  # Pre-normalization for better gradient flow
    ):
        super().__init__()

        # Embed each scalar parameter with larger capacity
        self.input_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Learnable positional encoding for each feature
        self.pos_encoding = nn.Parameter(torch.randn(1, n_params, d_model) * 0.02)

        # Use Pre-LN transformer for better gradient flow
        if use_prenorm:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-normalization
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Use CLS token instead of mean pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Deeper regression head with skip connection
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1),
        )

    def forward(self, x):
        # x: (batch, 19)
        batch_size = x.shape[0]

        x = x.unsqueeze(-1)               # (batch, 19, 1)
        x = self.input_embed(x)           # (batch, 19, d_model)

        # Add positional encoding to distinguish features
        x = x + self.pos_encoding        # (batch, 19, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 20, d_model)

        x = self.encoder(x)               # (batch, 20, d_model)

        # Use CLS token for prediction
        x = x[:, 0]                       # (batch, d_model)
        y = self.regressor(x)             # (batch, 1)
        return y


class PMSSMTransformerTabular(nn.Module):
    """
    Transformer designed specifically for tabular data.
    Instead of treating features as sequence tokens, this uses
    multi-head attention to learn feature interactions directly.
    """
    def __init__(
        self,
        n_params=19,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.0,
    ):
        super().__init__()

        # Individual feature embeddings (each feature gets its own embedding)
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
            ) for _ in range(n_params)
        ])

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Attention pooling instead of CLS token
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1),
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1),
        )

    def forward(self, x):
        # x: (batch, 19)
        batch_size = x.shape[0]

        # Embed each feature separately
        embedded = []
        for i, emb in enumerate(self.feature_embeddings):
            feat = x[:, i:i+1].unsqueeze(-1)  # (batch, 1, 1)
            embedded.append(emb(feat))  # (batch, 1, d_model)

        x = torch.cat(embedded, dim=1)  # (batch, 19, d_model)

        # Apply transformer
        x = self.encoder(x)  # (batch, 19, d_model)

        # Attention pooling
        attn_weights = self.attention_pool(x)  # (batch, 19, 1)
        x = (x * attn_weights).sum(dim=1)  # (batch, d_model)

        # Regress
        y = self.regressor(x)  # (batch, 1)
        return y
