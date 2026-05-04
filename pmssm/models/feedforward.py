"""
Feedforward MLP model for pMSSM regression.

This module provides a traditional multi-layer perceptron architecture
as a baseline comparison to transformer models.
"""

import torch.nn as nn


class PMSSMFeedForward(nn.Module):
    """
    Traditional feedforward neural network for pMSSM regression.

    Uses separate embedding for each input parameter followed by
    fully-connected layers. Dropout is applied after each ReLU so
    MC-Dropout inference can be used for uncertainty estimation.
    """
    def __init__(
        self,
        n_params=19,
        d_model=64,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.0,
    ):
        super().__init__()

        # Embed each scalar parameter
        self.input_embed = nn.Linear(1, d_model)

        # Build a stack of fully connected layers with dropout for MC sampling
        layers = []
        in_features = n_params * d_model
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, dim_feedforward))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_features = dim_feedforward
        self.fc_layers = nn.Sequential(*layers)

        # Output layer
        self.regressor = nn.Linear(dim_feedforward, 1)

    def forward(self, x):
        # x: (batch, n_params)
        x = x.unsqueeze(-1)              # (batch, n_params, 1)
        x = self.input_embed(x)          # (batch, n_params, d_model)
        x = x.flatten(start_dim=1)       # (batch, n_params * d_model)
        x = self.fc_layers(x)            # (batch, dim_feedforward)
        y = self.regressor(x)            # (batch, 1)
        return y
