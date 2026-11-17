"""Resolve the parameter budget and construct a regular overlapping leaf graph."""
from dataclasses import dataclass
from math import ceil, gcd, lcm, prod

import numpy as np

from ._common import ADDRESS as R, HIDDEN as H, WIDTH as D
from ._common import HAT_LEVELS, LINKS, SLOTS, SYMBOLS, integer


@dataclass(frozen=True)
class Shape:
    levels: int
    nodes: int
    facets: int
    live_nodes: int
    banks: int
    experts: int
    active_experts: int

    @property
    def degree(self):
        return gcd(self.nodes, 4)

    @property
    def leaf_links(self):
        return self.degree * self.experts // self.nodes

    @property
    def nominations(self):
        # With at most degree parents per ID this guarantees the requested
        # number of *distinct* experts, even in the worst possible overlap.
        return max(ceil(self.leaf_links / 4),
                   ceil(self.active_experts * min(self.degree, self.live_nodes) / self.live_nodes))

    def make_links(self, rng):
        groups = [rng.permutation(self.experts).reshape(-1, self.leaf_links)
                  for _ in range(self.degree)]
        return np.vstack(groups).astype(np.int32)


def weight_specs(s):
    """Map stored trainable arrays to (shape, initialization standard deviation)."""
    specs = {"byte": ((256, D), .35), "link": ((LINKS, D), .1),
             "slot": ((SLOTS, D), .12), "head": ((255, D), .11),
             "head_bias": ((255,), 0.), "context": ((R, D), D**-.5),
             "entry": ((s.nodes, R), .45), "final_gain": ((D,), 0.), "final_bias": ((D,), 0.)}
    for i in range(HAT_LEVELS):
        prefix = f"hat{i}."
        specs.update({prefix + name: (shape, scale) for name, shape, scale in (
            ("qkv", (3 * R, D), D**-.5), ("read", (D, R), .04),
            ("norm_gain", (D,), 0.), ("norm_bias", (D,), 0.),
            ("gates", (2, D), D**-.5), ("forget_in", (R, D), D**-.5),
            ("forget_out", (R, R), R**-.5), ("log_speed", (1,), 0.),
            ("dt_bias", (R,), .35), ("out_gate", (R, D), 0.),
            ("out_bias", (R,), 0.), ("rms_gain", (R,), 0.),
            ("query", (R, D), D**-.5), ("key", (SYMBOLS, R), .7),
            ("value", (SYMBOLS, D), .1))})
    for i in range(s.levels):
        prefix = f"mfs{i}."
        children = s.experts if i == s.levels - 1 else s.nodes
        specs.update({prefix + name: (shape, scale) for name, shape, scale in (
            ("node", (s.nodes, R), .27), ("scale", (s.nodes, s.facets, R), .3),
            ("shift", (s.nodes, s.facets, R), .32), ("mix", (s.facets, R), .45),
            ("mix_bias", (s.nodes, s.facets), .12), ("key", (children, R), .8))})
    per_bank = s.experts // s.banks
    specs.update({"up": ((s.banks, per_bank, H, D + R), .27),
                  "bias": ((s.banks, per_bank, H), 0.),
                  "down": ((s.banks, per_bank, D, H), .25)})
    return specs


def parameter_count(shape):
    return sum(prod(dims) for dims, _ in weight_specs(shape).values())


def resolve(*, parameters, levels, nodes, facets, live_nodes, banks, experts, active_experts):
    budget = 1_000_000 if parameters is None else integer(parameters, "parameters", 50_000, 100_000_000)
    depth = integer(levels, "levels", 1, 12)
    n = 64 if nodes is None else integer(nodes, "nodes", 1, 256)
    if experts is not None:
        experts = integer(experts, "experts", 1, 1_000_000)
        if nodes is None:
            n = gcd(n, experts)
    f = 16 if facets is None else integer(facets, "facets", 1, 32)
    b = integer(banks, "banks", 1, 64)
    per_expert = R + H * (D + R + 1 + D)
    while True:
        fixed = parameter_count(Shape(depth, n, f, 1, b, 0, 1))
        group = lcm(n, b)
        fit = (budget - fixed) // per_expert
        if experts is not None or fit >= group:
            break
        if nodes is None and n > 1:
            n //= 2
        elif facets is None and f > 1:
            f //= 2
        else:
            raise ValueError("parameter budget is too small for this topology")
    if experts is None:
        # Allocate complete sets of leaf/bank groups. The 1M profile resolves
        # to the historical 4,096 endpoints with uniform tiny MLPs.
        quantum = 4 * group if fit >= 32 * group else group
        e = fit // quantum * quantum
    else:
        e = experts
    if e % group:
        raise ValueError("experts must be divisible by both nodes and banks")
    if fixed + e * per_expert > (100_000_000 if parameters is None else budget):
        raise ValueError("expert banks exceed the parameter budget")
    live = min(16, max(1, n // 4)) if live_nodes is None else integer(live_nodes, "live_nodes", 1, n)
    degree = gcd(n, 4)
    guaranteed = min(e, live * (degree * e // n) // min(degree, live))
    active = (max(1, min(128, e // 32, guaranteed)) if active_experts is None
              else integer(active_experts, "active_experts", 1, guaranteed))
    return Shape(depth, n, f, live, b, e, active)
