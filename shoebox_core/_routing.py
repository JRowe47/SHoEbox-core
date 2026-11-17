"""Six-stage SHoE controller. Only live parents form facets or send content."""
from dataclasses import dataclass

import numpy as np

from . import _facets
from ._common import coalesce, softmax, top, update


def leaf_selection(outputs, mass, links, shape):
    nominees = top(outputs, shape.nominations)
    ids = np.take_along_axis(links, nominees, axis=1)
    contributions = mass[:, None] * np.take_along_axis(outputs, nominees, axis=1)
    candidates, inverse, counts = np.unique(ids.ravel(), return_inverse=True, return_counts=True)
    votes = np.zeros(len(candidates), np.float32)
    np.add.at(votes, inverse.ravel(), contributions.ravel())
    ranking = np.log(votes) + .25 * np.log1p(counts.astype(np.float32))
    selected = candidates[top(ranking, shape.active_experts)]

    # A nomination admits an ID to the final vote. Once selected, each of its
    # connected *live* parents contributes content, including non-nominators.
    order = np.argsort(selected)
    sorted_ids = selected[order]
    places = np.searchsorted(sorted_ids, links).clip(max=len(selected) - 1)
    parent, slot = np.nonzero(sorted_ids[places] == links)
    return selected, parent, slot, order[places[parent, slot]]


@dataclass
class Handoff:
    """Sparse edges to selected destinations, with path mass and parent content."""
    parent: np.ndarray
    slot: np.ndarray
    child: np.ndarray
    incoming_mass: np.ndarray
    weight: np.ndarray
    fraction: np.ndarray
    message: np.ndarray

    @classmethod
    def build(cls, state, mass, outputs, parent, slot, child, count):
        joint = mass[parent] * outputs[parent, slot]
        incoming = np.zeros(count, np.float32)
        np.add.at(incoming, child, joint)
        weight = incoming / incoming.sum()
        fraction = joint / incoming[child]
        message = np.zeros((count, state.shape[1]), np.float32)
        np.add.at(message, child, fraction[:, None] * state[parent])
        return cls(parent, slot, child, incoming, weight, fraction, message)

    def backward(self, state, mass, outputs, dweight, dmessage):
        # Differentiate both normalizations on the exact selected support.
        p, s, c = self.parent, self.slot, self.child
        dtotal = (dweight - dweight @ self.weight) / self.incoming_mass.sum()
        difference = state[p] - self.message[c]
        dedge = dtotal[c] + np.einsum("ij,ij->i", dmessage[c], difference) / self.incoming_mass[c]
        dout = np.zeros_like(outputs)
        dout[p, s] = mass[p] * dedge
        dmass = np.zeros_like(mass)
        np.add.at(dmass, p, outputs[p, s] * dedge)
        dstate = np.zeros_like(state)
        np.add.at(dstate, p, self.fraction[:, None] * dmessage[c])
        return dout, dmass, dstate


@dataclass
class Stage:
    ids: np.ndarray
    state: np.ndarray
    mass: np.ndarray
    links: np.ndarray | None
    outputs: np.ndarray
    facets: tuple
    handoff: Handoff


def route(weights, shape, graph, payload):
    context = np.tanh(weights["context"] @ payload)
    entry = weights["entry"] @ context
    ids = top(entry, shape.live_nodes)
    mass = softmax(entry[ids])
    message = np.zeros((len(ids), len(context)), np.float32)
    stages = []
    for level in range(shape.levels):
        prefix = f"mfs{level}."
        state = np.tanh(context + weights[prefix + "node"][ids] + message)
        queries = state[:, None, :] * weights[prefix + "scale"][ids] + weights[prefix + "shift"][ids]
        mixture = softmax(state @ weights[prefix + "mix"].T + weights[prefix + "mix_bias"][ids])
        links = graph[ids] if level == shape.levels - 1 else None
        keys = weights[prefix + "key"] if links is None else weights[prefix + "key"][links]
        outputs, facets = _facets.forward(queries, keys, mixture)
        if links is None:
            selected = top(mass @ outputs, shape.live_nodes)
            parent = np.repeat(np.arange(len(ids)), len(selected))
            slot = np.tile(selected, len(ids))
            child = np.tile(np.arange(len(selected)), len(ids))
        else:
            selected, parent, slot, child = leaf_selection(outputs, mass, links, shape)
        handoff = Handoff.build(state, mass, outputs, parent, slot, child, len(selected))
        stages.append(Stage(ids, state, mass, links, outputs, facets, handoff))
        ids, mass, message = selected, handoff.weight, handoff.message
    return ids, mass, message, (context, stages)


def backward(weights, payload, tape, dmass, dmessage, rate):
    context, stages = tape
    dcontext = np.zeros_like(context)
    for level in reversed(range(len(stages))):
        prefix = f"mfs{level}."
        stage = stages[level]
        ids, state = stage.ids, stage.state
        dout, dmass, dstate = stage.handoff.backward(state, stage.mass, stage.outputs, dmass, dmessage)
        dq, dk, dmix = _facets.backward(stage.facets, dout)
        dstate += (dq * weights[prefix + "scale"][ids]).sum(axis=1)
        dstate += dmix @ weights[prefix + "mix"]
        dmessage = dstate * (1 - state * state)
        dcontext += dmessage.sum(axis=0)

        update(weights[prefix + "node"], dmessage, rate, ids)
        update(weights[prefix + "mix"], dmix.T @ state, rate)
        update(weights[prefix + "mix_bias"], dmix, rate, ids)
        update(weights[prefix + "scale"], dq * state[:, None, :], rate, ids)
        update(weights[prefix + "shift"], dq, rate, ids)
        if stage.links is None:
            update(weights[prefix + "key"], dk, rate)
        elif rate:
            key_ids, key_grad = coalesce(stage.links, dk)
            update(weights[prefix + "key"], key_grad, rate, key_ids)

    first = stages[0]
    dentry = first.mass * (dmass - dmass @ first.mass)
    dcontext += dentry @ weights["entry"][first.ids]
    dcontext *= 1 - context * context
    dpayload = dcontext @ weights["context"]
    update(weights["entry"], np.outer(dentry, context), rate, first.ids)
    update(weights["context"], np.outer(dcontext, payload), rate)
    return dpayload
