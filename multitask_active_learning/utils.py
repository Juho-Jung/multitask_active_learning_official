"""Utility functions for multi-task active learning."""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def normalize_scores(scores: torch.Tensor, method: str = 'minmax') -> torch.Tensor:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: Raw scores to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')

    Returns:
        Normalized scores in [0, 1] range
    """
    if len(scores) == 0:
        return scores

    if method == 'minmax':
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-8:
            return torch.zeros_like(scores)

        normalized = (scores - min_score) / (max_score - min_score)
        return torch.clamp(normalized, 0.0, 1.0)

    elif method == 'zscore':
        mean_score = scores.mean()
        std_score = scores.std()

        if std_score < 1e-8:
            return torch.zeros_like(scores)

        normalized = (scores - mean_score) / std_score
        # Convert to [0, 1] using sigmoid
        return torch.sigmoid(normalized)

    elif method == 'robust':
        median_score = scores.median()
        mad = torch.median(torch.abs(scores - median_score))

        if mad < 1e-8:
            return torch.zeros_like(scores)

        normalized = (scores - median_score) / (1.4826 * mad)  # 1.4826 for consistency with std
        return torch.sigmoid(normalized)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of logits.

    Args:
        logits: Input logits
        dim: Dimension to compute entropy over

    Returns:
        Entropy values
    """
    probs = F.softmax(logits, dim=dim)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
    return entropy


def compute_margin(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute margin (difference between top-2 predictions).

    Args:
        logits: Input logits
        dim: Dimension to compute margin over

    Returns:
        Margin values
    """
    probs = F.softmax(logits, dim=dim)
    sorted_probs, _ = torch.sort(probs, dim=dim, descending=True)
    margin = sorted_probs[..., 0] - sorted_probs[..., 1]
    return margin
