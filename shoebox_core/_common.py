"""Dimensions and sparse online learning primitives."""
import numpy as np

WIDTH, ADDRESS, HIDDEN = 28, 8, 3
SYMBOLS, HAT_LEVELS, LINKS, SLOTS = 64, 2, 8, 256


def integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def top(scores, count):
    # Stable sort also fixes tie behaviour by candidate order after a restart.
    return np.argsort(-scores, axis=-1, kind="stable")[..., :count]


def update(weights, grad, rate, ids=None):
    if rate == 0:
        return
    delta = rate * np.clip(grad, -2., 2.)
    if ids is None:
        weights -= delta
    else:
        # Callers supply unique rows. Converging edges are reduced beforehand.
        weights[ids] -= delta


def weighted_rows(weights, rows):
    return np.einsum("i,ij->j", weights, rows)


def coalesce(ids, values):
    unique, inverse = np.unique(ids.ravel(), return_inverse=True)
    rows = np.zeros((len(unique), values.shape[-1]), np.float32)
    np.add.at(rows, inverse.ravel(), values.reshape(-1, values.shape[-1]))
    return unique, rows
