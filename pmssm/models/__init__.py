"""
Neural network model architectures for pMSSM regression.

This module exports all model classes and utility functions for model
introspection and naming.
"""

from .transformer import PMSSMTransformer, PMSSMTransformerTabular
from .feedforward import PMSSMFeedForward

import torch.nn as nn

__all__ = [
    'PMSSMTransformer',
    'PMSSMTransformerTabular',
    'PMSSMFeedForward',
    'is_transformer',
    'get_model_name',
]


def is_transformer(model: nn.Module) -> bool:
    """Check if a model uses transformer architecture."""
    return any(
        isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer))
        for m in model.modules()
    )


def get_model_name(model: nn.Module) -> str:
    """Get a clean name for the model for use in plot filenames."""
    class_name = model.__class__.__name__
    if class_name == "PMSSMTransformer":
        return "transformer"
    elif class_name == "PMSSMTransformerTabular":
        return "transformer_tabular"
    elif class_name == "PMSSMFeedForward":
        return "MLP"
    else:
        return "transformer" if is_transformer(model) else "MLP"
