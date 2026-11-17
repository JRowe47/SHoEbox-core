"""Physical MLP banks shared by overlapping leaf support maps."""
from typing import NamedTuple

import numpy as np

from ._common import WIDTH, update, weighted_rows


class Dispatch(NamedTuple):
    address: tuple
    inputs: np.ndarray
    up: np.ndarray
    down: np.ndarray
    hidden: np.ndarray
    output: np.ndarray
    gain: np.float32


def execute(weights, shape, payload, ids, gates, messages):
    address = np.divmod(ids, shape.experts // shape.banks)
    # Keep the entire current HAT payload as a distinct expert input field.
    inputs = np.concatenate((np.broadcast_to(payload, (len(ids), len(payload))), messages), axis=1)
    up, down = weights["up"][address], weights["down"][address]
    hidden = np.tanh(np.einsum("ehi,ei->eh", up, inputs) + weights["bias"][address])
    output = np.einsum("eoh,eh->eo", down, hidden)
    gain = np.float32(len(ids)**.5)
    features = payload + gain * weighted_rows(gates, output)
    return features, Dispatch(address, inputs, up, down, hidden, output, gain)


def backward(weights, tape, gates, grad, rate):
    address, inputs, up, down, hidden, output, gain = tape
    dgates = gain * (output @ grad)
    dout = (gain * gates)[:, None] * grad
    dh = np.einsum("eo,eoh->eh", dout, down) * (1 - hidden * hidden)
    dx = np.einsum("eh,ehi->ei", dh, up)
    update(weights["down"], dout[:, :, None] * hidden[:, None, :], rate, address)
    update(weights["up"], dh[:, :, None] * inputs[:, None, :], rate, address)
    update(weights["bias"], dh, rate, address)
    return grad + dx[:, :WIDTH].sum(axis=0), dgates, dx[:, WIDTH:]
