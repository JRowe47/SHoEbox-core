"""Streaming byte model with HAT, deep multifacet routing and sparse expert banks."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from . import _experts, _facets, _hat, _routing
from ._common import ADDRESS, HAT_LEVELS, LINKS, SLOTS, SYMBOLS, WIDTH, integer, update
from ._shape import Shape, parameter_count, resolve, weight_specs
from .events import ByteEvents, as_bytes

_RATE_NAMES = ("fast_rate", "query_rate", "route_rate", "head_rate")
_MAGIC = b"SHOECR02"
# For a leaf with index 256+byte in a one-based binary heap, its eight
# internal ancestors are the successive right shifts. Store zero-based rows.
_LEAVES = 256 + np.arange(256)[:, None]
_PATHS = (_LEAVES >> np.arange(8, 0, -1)) - 1
_SIGNS = (2 * ((_LEAVES >> np.arange(7, -1, -1)) & 1) - 1).astype(np.float32)


def _events(data):
    return data if isinstance(data, ByteEvents) else ByteEvents.static(data)


def _empty_context(s):
    return {
        "memory": np.zeros((LINKS, HAT_LEVELS, ADDRESS, ADDRESS), np.float32),
        "clocks": np.zeros(LINKS, np.uint64), "features": np.zeros(WIDTH, np.float32),
        "bank_calls": np.zeros(s.banks, np.uint64),
        "address": np.full(3, -1, np.int32),
        "symbols": np.full((HAT_LEVELS, 2), -1, np.int32),
        "nodes": np.full((s.levels, s.live_nodes), -1, np.int32),
        "node_weights": np.zeros((s.levels, s.live_nodes), np.float32),
        "experts": np.full(s.active_experts, -1, np.int32),
        "expert_weights": np.zeros(s.active_experts, np.float32),
    }


class Model:
    """One mutable stream with immediate next-byte SGD and bounded HAT memory.

    The default is the six-level MFS preset. An explicit expert count fixes the
    physical bank capacity. Otherwise a nominal parameter budget sizes it.
    """

    def __init__(self, *, parameters=None, seed=7, levels=6, nodes=None, facets=None,
                 live_nodes=None, banks=4, experts=None, active_experts=None):
        seed = integer(seed, "seed", 0, 2**32 - 1)
        self._shape = resolve(parameters=parameters, levels=levels, nodes=nodes, facets=facets,
                              live_nodes=live_nodes, banks=banks, experts=experts,
                              active_experts=active_experts)
        _facets._library()
        rng = np.random.default_rng(seed)
        self._weights = {}
        for name, (dims, std) in weight_specs(self._shape).items():
            values = rng.standard_normal(dims, dtype=np.float32) * std if std else np.zeros(dims, np.float32)
            if name.endswith((".scale", "_gain")):
                values += 1
            elif name.endswith(".out_bias"):
                values.fill(3.)
            elif name.endswith(".dt_bias"):
                values -= 3.7
            elif name == "hat0.log_speed":
                values.fill(np.log(4.))
            self._weights[name] = values
        self._graph = self._shape.make_links(rng)
        self._rates = np.array([.04, .025, .02, .16], np.float32)
        self._training = True
        self._state = _empty_context(self._shape)

    def _alive(self):
        if self._weights is None:
            raise RuntimeError("model has been closed")

    def __enter__(self):
        self._alive()
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._weights = self._graph = self._state = None

    def _forward(self, byte, link, slot):
        w, s, shape = self._weights, self._state, self._shape
        payload, symbols = _hat.encode(w, s["memory"], byte, link, slot)
        ids, gates, messages, route = _routing.route(w, shape, self._graph, payload)
        features, experts = _experts.execute(w, shape, payload, ids, gates, messages)
        s["features"] = features
        s["clocks"][link] += 1
        s["bank_calls"] += np.bincount(experts.address[0], minlength=shape.banks).astype(np.uint64)
        s["address"][:] = byte, link, slot
        for i, symbol in enumerate(symbols.steps):
            s["symbols"][i] = symbol.symbols
        for i, stage in enumerate(route[1]):
            s["nodes"][i] = stage.ids
            s["node_weights"][i] = stage.mass
        s["experts"][:] = ids
        s["expert_weights"][:] = gates
        return payload, symbols, route, experts

    def _score(self, target, tape):
        w, s = self._weights, self._state
        ids, signs = _PATHS[target], _SIGNS[target]
        features = s["features"]
        head = w["head"][ids]
        logits = head @ features + w["head_bias"][ids]
        errors = np.logaddexp(0., -signs * logits)
        if self._training:
            fast, query, routing, head_rate = self._rates
            grad = -signs * np.exp(-np.logaddexp(0., signs * logits))
            dfeatures = grad @ head
            update(w["head"], np.outer(grad, features), head_rate / (1 + features @ features), ids)
            update(w["head_bias"], grad, head_rate * .05, ids)
            payload, symbols, route, experts = tape
            dp, dm, dmsg = _experts.backward(w, experts, s["expert_weights"], dfeatures, fast)
            dp += _routing.backward(w, payload, route, dm, dmsg, routing)
            byte, link, slot = s["address"]
            _hat.backward(w, symbols, dp, byte, link, slot, query)
        return float(errors.sum())

    def run(self, inputs, targets):
        """Consume aligned input events and target bytes, returning mean NLL."""
        self._alive()
        events = _events(inputs)
        target = targets.values if isinstance(targets, ByteEvents) else as_bytes(targets)
        if not target or len(events.values) != len(target):
            raise ValueError("inputs and targets must have the same nonzero length")
        loss = 0.
        for byte, link, slot, y in zip(events.values, events.links, events.slots, target):
            tape = self._forward(byte, link, slot)
            loss += self._score(y, tape)
        return loss / len(target)

    def train(self, data):
        """Train on adjacent byte pairs. Use run for aligned streaming chunks."""
        self._alive()
        if not self._training:
            raise RuntimeError("this is a frozen inference model")
        events = _events(data)
        return self.run(events[:-1], events.values[1:])

    def logits(self):
        """Return normalized natural log probabilities for all 256 next bytes."""
        self._alive()
        z = self._weights["head"] @ self._state["features"] + self._weights["head_bias"]
        return -np.logaddexp(0., -_SIGNS * z[_PATHS]).sum(axis=1)

    def observe(self, byte, *, link=0, slot=None):
        """Advance a frozen model by one addressed event and return predictions."""
        self._alive()
        if self._training:
            raise RuntimeError("observe requires an inference copy")
        byte = integer(byte, "byte", 0, 255)
        link = integer(link, "link", 0, LINKS - 1)
        slot = byte if slot is None and link == 0 else integer(slot, "slot", 0, SLOTS - 1)
        if link == 0 and slot != byte:
            raise ValueError("static dictionary slot must equal the byte")
        self._forward(byte, link, slot)
        return self.logits()

    def clone(self):
        self._alive()
        return deepcopy(self)

    def inference_copy(self, *, cold=True):
        model = self.clone()
        model._training = False
        if cold:
            model.reset_context()
        return model

    def evaluate(self, data):
        events = _events(data)
        with self.inference_copy() as frozen:
            return frozen.run(events[:-1], events.values[1:])

    def generate(self, prompt, *, length=256, temperature=.8, seed=7):
        """Return prompt bytes and a sampled continuation from a cold frozen copy."""
        prompt = _events(prompt)
        length = integer(length, "length", 0, 2**63 - 1)
        if not prompt.values or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("a nonempty prompt and finite positive temperature are required")
        result = bytearray(prompt.values)
        rng = np.random.default_rng(seed)
        with self.inference_copy() as frozen:
            for byte, link, slot in zip(prompt.values, prompt.links, prompt.slots):
                frozen._forward(byte, link, slot)
            for _ in range(length):
                z = frozen.logits().astype(np.float64)
                probs = np.exp((z - z.max()) / temperature)
                byte = int(rng.choice(256, p=probs / probs.sum()))
                result.append(byte)
                frozen._forward(byte, 0, byte)
        return bytes(result)

    def finish(self):
        """All credit has already been applied when run() returns."""
        self._alive()

    def reset_context(self):
        self._alive()
        self._state = _empty_context(self._shape)

    def configure(self, *, fast_rate=None, query_rate=None, route_rate=None, head_rate=None):
        """Set learning rates for expert MLPs, HAT, MFS and byte readout."""
        self._alive()
        values = (fast_rate, query_rate, route_rate, head_rate)
        rates = [old if new is None else new for old, new in zip(self._rates, values)]
        if any(not np.isfinite(v) or v < 0 or v > np.finfo(np.float32).max for v in rates):
            raise ValueError("rates must be finite nonnegative float32 values")
        self._rates[:] = rates

    @property
    def parameter_count(self):
        self._alive()
        return parameter_count(self._shape)

    def stats(self):
        self._alive()
        s = self._shape
        events = sum(map(int, self._state["clocks"]))
        return dict(architecture="hat-mfs-shoe", checkpoint_version="shoebox-core-2",
                    parameters=self.parameter_count, bytes=events, training=self._training,
                    **asdict(s), level_kinds=["mfs"] * s.levels, events=[events] * s.levels,
                    expert_calls=self._state["bank_calls"].tolist(), parent_degree=s.degree,
                    leaf_links=s.leaf_links, nominations=s.nominations, hat_levels=HAT_LEVELS,
                    hat_symbols=SYMBOLS, symbols_per_level=2, link_clocks=self._state["clocks"].tolist(),
                    learning_rates=dict(zip(_RATE_NAMES, self._rates.tolist())))

    def last_route(self):
        """Return a snapshot of the last occurrence's HAT, controller and expert IDs."""
        self._alive()
        state = self._state
        if state["address"][0] < 0:
            return None
        byte, link, slot = map(int, state["address"])
        bank_size = self._shape.experts // self._shape.banks
        return dict(byte=byte, link=link, slot=slot, symbols=state["symbols"].copy(),
                    nodes=state["nodes"].copy(), node_weights=state["node_weights"].copy(),
                    experts=state["experts"].copy(), expert_weights=state["expert_weights"].copy(),
                    bank_addresses=np.stack(np.divmod(state["experts"], bank_size), axis=-1))

    def dumps(self):
        """Serialize parameters, topology, rates, stream state and last route."""
        self._alive()
        out = BytesIO()
        arrays = {"w." + k: v for k, v in self._weights.items()}
        arrays.update({"s." + k: v for k, v in self._state.items()})
        np.savez(out, topology=np.array(list(asdict(self._shape).values()), dtype=np.int32),
                 links=self._graph, rates=self._rates, training=np.array(self._training), **arrays)
        raw = out.getvalue()
        return _MAGIC + hashlib.sha256(raw).digest() + raw

    @classmethod
    def loads(cls, data):
        data = bytes(data)
        if len(data) < 40 or hashlib.sha256(data[40:]).digest() != data[8:40]:
            raise ValueError("invalid or corrupted SHoEbox-core checkpoint")
        if data[:8] == b"SHOECR01":
            raise ValueError("SHOECR01 uses the v0.1 summarizer topology. Load it with SHoEbox-core 647e320.")
        if data[:8] != _MAGIC:
            raise ValueError("unknown SHoEbox-core checkpoint format")
        try:
            with np.load(BytesIO(data[40:]), allow_pickle=False) as archive:
                model = cls._restore(archive)
        except (BadZipFile, EOFError, KeyError, OSError, TypeError) as exc:
            raise ValueError("invalid SHoEbox-core checkpoint contents") from exc
        return model

    @classmethod
    def _restore(cls, archive):
        def array(name, dtype, shape):
            value = archive[name]
            if value.dtype != dtype or value.shape != shape:
                raise ValueError(f"invalid checkpoint array {name}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"nonfinite checkpoint array {name}")
            return value

        names = [field.name for field in fields(Shape)]
        config = dict(zip(names, array("topology", np.int32, (len(names),))))
        model = cls.__new__(cls)
        shape = model._shape = resolve(parameters=None, **config)
        model._weights = {k: array("w." + k, np.float32, dims)
                          for k, (dims, _) in weight_specs(shape).items()}
        model._state = {k: array("s." + k, v.dtype, v.shape)
                        for k, v in _empty_context(shape).items()}
        graph = model._graph = array("links", np.int32, (shape.nodes, shape.leaf_links))
        if (np.any(graph < 0) or np.any(graph >= shape.experts)
                or np.any(np.diff(np.sort(graph, axis=1), axis=1) == 0)
                or np.any(np.bincount(graph.ravel(), minlength=shape.experts) != shape.degree)):
            raise ValueError("invalid checkpoint leaf graph")
        model._rates = array("rates", np.float32, (len(_RATE_NAMES),))
        if np.any(model._rates < 0):
            raise ValueError("invalid checkpoint rates")
        model._training = bool(array("training", np.bool_, ()))
        _facets._library()
        return model

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(self.dumps())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        return cls.loads(Path(path).read_bytes())
