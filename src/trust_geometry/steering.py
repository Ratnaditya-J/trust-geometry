"""Direction construction and residual-stream interventions for R1."""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("direction has zero or non-finite norm")
    return vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def ordinal_direction(centroids: np.ndarray, roles: list[str], order: list[str]) -> np.ndarray:
    """Minimum-norm axis mapping centroids to the frozen high-to-low role scores."""
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.shape[0] != len(roles) or set(roles) != set(order):
        raise ValueError("centroid/role/order mismatch")
    center = centroids - centroids.mean(0, keepdims=True)
    midpoint = (len(order) - 1) / 2
    scores = np.array([midpoint - order.index(role) for role in roles], dtype=np.float64)
    direction, *_ = np.linalg.lstsq(center, scores, rcond=None)
    return unit(direction)


def pc1_direction(centroids: np.ndarray, roles: list[str], high="system", low="tool") -> np.ndarray:
    center = np.asarray(centroids, dtype=np.float64) - np.mean(centroids, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(center, full_matrices=False)
    direction = unit(vt[0])
    hi, lo = roles.index(high), roles.index(low)
    if np.dot(centroids[hi] - centroids[lo], direction) < 0:
        direction = -direction
    return direction


def mean_difference(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return unit(np.asarray(positive).mean(0) - np.asarray(negative).mean(0))


def orthogonalize(direction: np.ndarray, nuisances: list[np.ndarray]) -> np.ndarray:
    out = unit(direction).copy()
    if not nuisances:
        return out
    basis = np.stack([unit(n) for n in nuisances], axis=1)
    q, _ = np.linalg.qr(basis)
    out -= q @ (q.T @ out)
    return unit(out)


def projection_scale(activations: np.ndarray, direction: np.ndarray) -> float:
    scale = float(np.std(np.asarray(activations) @ unit(direction)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("projected activation scale is degenerate")
    return scale


def hidden_index_to_decoder_layer(hidden_index: int) -> int:
    """HF hidden_states[0] is embeddings; hidden_states[k] is layer k-1 output."""
    if hidden_index < 1:
        raise ValueError("cannot intervene on the embedding entry with a decoder-layer hook")
    return hidden_index - 1


@contextmanager
def residual_steer(model, hidden_index: int, direction: np.ndarray, magnitude: float,
                   positions: tuple[int, int] | None = None):
    """Add a vector to one decoder-layer output, optionally only on [start, end)."""
    import torch

    layer_index = hidden_index_to_decoder_layer(hidden_index)
    vector = torch.as_tensor(np.asarray(direction).copy(), device=model.device, dtype=model.dtype) * magnitude

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        changed = tensor.clone()
        if positions is None:
            changed += vector
        else:
            start, end = positions
            changed[:, start:min(end, changed.shape[1]), :] += vector
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    handle = model.model.layers[layer_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
