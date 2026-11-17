"""Source-indexed HAT symbols with a one-head tokenwise minKron recurrence."""
from typing import NamedTuple

import numpy as np

from ._common import ADDRESS as R, HAT_LEVELS, SYMBOLS, softmax, top, update, weighted_rows


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def unit(x):
    length = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x / np.maximum(length, 1e-12), length


def unit_backward(direction, length, grad):
    projection = direction * np.sum(direction * grad, axis=-1, keepdims=True)
    return (grad - np.where(length > 1e-12, projection, 0.)) / np.maximum(length, 1e-12)


def layer_norm(x, gain, bias):
    centered = x - x.mean()
    inv = 1 / np.sqrt(np.mean(centered * centered) + 1e-5)
    norm = centered * inv
    return norm * gain + bias, (norm, inv)


def norm_backward(tape, gain, grad):
    norm, inv = tape
    dnorm = grad * gain
    return inv * (dnorm - dnorm.mean() - norm * (dnorm * norm).mean())


class MemoryStep(NamedTuple):
    raw: np.ndarray
    silu_gate: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray
    gates: np.ndarray
    forget_input: np.ndarray
    forget_logit: np.ndarray
    speed: np.float32
    log_decay: np.ndarray
    faded: np.ndarray
    score: np.float32
    value_norm: np.float32
    denom: np.float32
    coefficient: np.float32
    matrix: np.ndarray
    norm: np.ndarray
    inv_rms: np.float32
    out_gate: np.ndarray


def memory_forward(w, p, x, previous):
    raw = (w[p + "qkv"] @ x).reshape(3, R)
    silu_gate = sigmoid(raw)
    dirs, lengths = unit(raw * silu_gate)
    q, k, v = dirs
    gates = sigmoid(w[p + "gates"] @ x)
    erase, write = gates
    forget_input = w[p + "forget_in"] @ x
    forget_logit = w[p + "forget_out"] @ forget_input + w[p + "dt_bias"]
    speed = np.exp(w[p + "log_speed"][0])
    log_decay = -speed * np.logaddexp(0., forget_logit)
    faded = np.exp(log_decay)[:, None] * previous
    score = k @ faded @ v
    value_norm = v @ v
    denom = np.maximum(value_norm, 1e-18)
    coef = (write * value_norm - erase * score) / denom
    matrix = faded + coef * np.outer(k, v)
    read = (q @ matrix) * R**-.5
    inv_rms = 1 / np.sqrt((read @ read) / R + 1e-5)
    norm = read * inv_rms
    out_gate = sigmoid(w[p + "out_gate"] @ x + w[p + "out_bias"])
    result = norm * w[p + "rms_gain"] * out_gate
    tape = MemoryStep(raw, silu_gate, dirs, lengths, gates, forget_input, forget_logit,
                      speed, log_decay, faded, score, value_norm, denom, coef,
                      matrix, norm, inv_rms, out_gate)
    return result, tape


def memory_backward(w, p, x, t, grad, rate):
    q, k, v = t.directions
    erase, write = t.gates
    dout_gate = grad * t.norm * w[p + "rms_gain"] * t.out_gate * (1 - t.out_gate)
    dnorm = grad * w[p + "rms_gain"] * t.out_gate
    dread = t.inv_rms * (dnorm - t.norm * ((dnorm @ t.norm) / R)) * R**-.5
    dmatrix = np.outer(q, dread)
    dq = t.matrix @ dread
    dcoef = k @ dmatrix @ v
    dnumerator = dcoef / t.denom
    dscore = -erase * dnumerator
    dvalue_norm = write * dnumerator
    if t.value_norm > 1e-18:
        dvalue_norm -= dnumerator * t.coefficient
    dk = t.coefficient * (dmatrix @ v) + dscore * (t.faded @ v)
    dv = t.coefficient * (k @ dmatrix) + dscore * (k @ t.faded) + 2 * dvalue_norm * v
    draw = unit_backward(t.directions, t.lengths, np.stack((dq, dk, dv)))
    draw *= t.silu_gate * (1 + t.raw * (1 - t.silu_gate))
    draw = draw.ravel()
    dgates = dnumerator * np.array([-t.score, t.value_norm]) * t.gates * (1 - t.gates)
    dfaded = dmatrix + dscore * np.outer(k, v)
    dlog_decay = (dfaded * t.faded).sum(axis=1)
    dforget = -t.speed * dlog_decay * sigmoid(t.forget_logit)
    dforget_input = dforget @ w[p + "forget_out"]
    dx = (draw @ w[p + "qkv"] + dgates @ w[p + "gates"]
          + dforget_input @ w[p + "forget_in"] + dout_gate @ w[p + "out_gate"])

    update(w[p + "qkv"], np.outer(draw, x), rate)
    update(w[p + "gates"], np.outer(dgates, x), rate)
    update(w[p + "dt_bias"], dforget, rate)
    update(w[p + "log_speed"], dlog_decay @ t.log_decay, rate)
    update(w[p + "forget_out"], np.outer(dforget, t.forget_input), rate)
    update(w[p + "forget_in"], np.outer(dforget_input, x), rate)
    update(w[p + "rms_gain"], grad * t.norm * t.out_gate, rate)
    update(w[p + "out_gate"], np.outer(dout_gate, x), rate)
    update(w[p + "out_bias"], dout_gate, rate)
    return dx


class SymbolStep(NamedTuple):
    input_norm: tuple
    memory_input: np.ndarray
    memory: MemoryStep
    read: np.ndarray
    contextual_norm: tuple
    query_input: np.ndarray
    query: np.ndarray
    query_length: np.ndarray
    keys: np.ndarray
    key_lengths: np.ndarray
    probabilities: np.ndarray
    symbols: np.ndarray
    gain: np.ndarray


class HATTape(NamedTuple):
    steps: list
    final_norm: tuple


SYMBOL_GAIN = np.float32(SYMBOLS / (2 * np.sqrt(2.)))


def encode(w, memory, byte, link, slot):
    x = w["byte"][byte] + w["link"][link] + w["slot"][slot]
    steps = []
    for i in range(HAT_LEVELS):
        p = f"hat{i}."
        z, input_norm = layer_norm(x, w[p + "norm_gain"], w[p + "norm_bias"])
        read, mt = memory_forward(w, p, z, memory[link, i])
        memory[link, i] = mt.matrix
        h = x + w[p + "read"] @ read
        query_input, contextual_norm = layer_norm(h, w[p + "norm_gain"], w[p + "norm_bias"])
        query, query_length = unit(w[p + "query"] @ query_input)
        keys, key_lengths = unit(w[p + "key"])
        probs = softmax(4 * (keys @ query))
        ids = top(probs, 2)
        gain = probs[ids] * SYMBOL_GAIN
        steps.append(SymbolStep(input_norm, z, mt, read, contextual_norm, query_input,
                                query, query_length, keys, key_lengths, probs, ids, gain))
        x = h + weighted_rows(gain, w[p + "value"][ids])
    payload, final = layer_norm(x, w["final_gain"], w["final_bias"])
    return payload, HATTape(steps, final)


def backward(w, tape, grad, byte, link, slot, rate):
    dx = norm_backward(tape.final_norm, w["final_gain"], grad)
    update(w["final_gain"], tape.final_norm[0] * grad, rate)
    update(w["final_bias"], grad, rate)
    for i in reversed(range(HAT_LEVELS)):
        p = f"hat{i}."
        t = tape.steps[i]
        dprob = np.zeros_like(t.probabilities)
        dprob[t.symbols] = (w[p + "value"][t.symbols] @ dx) * SYMBOL_GAIN
        dscore = 4 * t.probabilities * (dprob - t.probabilities @ dprob)
        dq = unit_backward(t.query, t.query_length, dscore @ t.keys)
        dk = unit_backward(t.keys, t.key_lengths, np.outer(dscore, t.query))
        dzq = dq @ w[p + "query"]
        dh = dx + norm_backward(t.contextual_norm, w[p + "norm_gain"], dzq)
        dmem = dh @ w[p + "read"]
        dzm = memory_backward(w, p, t.memory_input, t.memory, dmem, rate)
        dinput = dh + norm_backward(t.input_norm, w[p + "norm_gain"], dzm)

        update(w[p + "value"], t.gain[:, None] * dx, rate, t.symbols)
        update(w[p + "key"], dk, rate)
        update(w[p + "query"], np.outer(dq, t.query_input), rate)
        update(w[p + "read"], np.outer(dh, t.read), rate)
        update(w[p + "norm_gain"], dzq * t.contextual_norm[0] + dzm * t.input_norm[0], rate)
        update(w[p + "norm_bias"], dzq + dzm, rate)
        dx = dinput
    for name, idx in (("byte", byte), ("link", link), ("slot", slot)):
        update(w[name], dx, rate, idx)
