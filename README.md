# SHoEbox-core

SHoEbox-core is a causal byte model built from HAT, a deep multifacet softmax (MFS) hierarchy and sparse MLP expert banks. Every byte produces one HAT occurrence, traverses every routing level and executes one set of leaf experts. Version 0.2 removes the four-byte grouping clocks and the upper summarizers.

## Default model

| Component | Stored | Active per input byte |
| --- | --- | --- |
| Hierarchical AutoTokenizer (HAT) | 2 codebooks of 64 symbols | 2 symbols per codebook |
| SHoE router | 6 levels, 64 nodes per level | 16 nodes on each level |
| Multifacet routing | 16 facets per node | All 16 facets of each live node |
| Terminal leaf links | 256 experts per leaf, 4 leaf parents per expert | Each live leaf nominates 64 experts |
| Physical expert banks | 4 banks of 1,024 MLPs | 128 distinct MLPs across the banks |
| Byte decoder | 255 binary decisions cover all 256 bytes | 8 decisions for an observed target |

The default has **973,229 trainable parameters**. Each expert sees the complete 28-value HAT payload and its own 8-value routing context. Its three hidden units write a residual into the payload before byte prediction. Each expert has a stable bank and row address. Several leaves can reach the same physical weights.

HAT carries two 8 × 8 recurrent matrices per input link. Each level runs one minKron head with normalized query, key and value vectors, learned coordinate-wise forgetting, and separate erase and write gates. It reads the updated matrix and selects its contextual symbols by cosine score. Link 0 is the immutable 256-byte dictionary. Live links 1–7 distinguish producer streams, with 256 slots each. A byte from one link advances only that link's HAT memory. The experts are stateless.

## Routing and learning

1. HAT supplies a contextual payload to both the router and the expert inputs. The entry gate chooses the first live nodes.
2. Each live node composes its own learned features with the payload's routing context and incoming parent features. Each facet produces a softmax over its legal destinations. A learned mixture of those distributions supplies the node's votes.
3. A destination receives the combined mass of its incoming edges and a content message weighted by those edges. The controller keeps the strongest nodes and renormalizes their mass at each depth.
4. The last level scores only its connected expert addresses. Leaf nominations and overlap determine the winning distinct IDs. Each winner receives content from every connected live parent and executes once. Its weight in the output is proportional to its incoming mass.
5. Online next-byte SGD updates the eight decoder decisions, executed MLPs, MFS gates and HAT. Derivatives flow through the selected gates, parent content and facet softmaxes on the current event. Hard top-k identities and the previous event's HAT matrices are detached. The current matrix read and write both receive task credit.

The architecture follows the one-pass topology in the October 30 `MFS_6_Level_64x16_Source_and_Precision.zip` and the static/live slot and contextual symbol interface in the earlier SHoE implementation. This core uses one minKron head per HAT level, a fused additive byte/address embedding and small, stateless MLP experts. [Earlier SHoE and HAT architecture](https://github.com/JRowe47/Sparse-Hierarchy-of-Experts/blob/b4bb7cfea18c3833819f6ed08973fd014f15f107/docs/architecture.md)

## Install and use

Requires Python 3.10+, NumPy 1.26+ and a C++17 compiler on Linux or WSL. The package builds its FP32 multifacet forward and backward kernels on first use. Set `CXX` to select a compiler and `SHOEBOX_CORE_CACHE` to select the build cache.

```bash
python -m pip install .
python -m unittest discover -s tests -v
```

```python
from shoebox_core import Model

with Model(seed=7) as model:
    loss = model.train(b"One observed byte sequence.\n")
    print(model.evaluate(b"A separate byte sequence.\n"))  # nats per byte
    print(model.last_route()["nodes"].shape)                # (6, 16)
    print(model.last_route()["bank_addresses"].shape)       # (128, 2)
    model.save("model.shoe")

with Model.load("model.shoe") as model:
    print(model.generate("One ", length=40))
```

`train(data)` consumes `len(data)-1` adjacent pairs. For a continuous stream, retain the boundary pair by feeding aligned slices to `run`.

```python
data = b"A continuous training stream with every boundary pair."
with Model() as model:
    for start in range(0, len(data) - 1, 16):
        stop = min(start + 16, len(data) - 1)
        model.run(data[start:stop], data[start + 1:stop + 1])
```

Pass text, bytes, a 1-D NumPy `uint8` array or addressed `ByteEvents`. `SlotDictionary` materializes immutable input snapshots. The caller owns its live publications.

```python
from shoebox_core import ByteEvents, Model, SlotDictionary

slots = SlotDictionary()
slots.publish(2, 7, b"AB")
events = slots.events([(0, 65), (2, 7)])  # A, A, B with their source addresses
slots.publish(2, 7, b"Z")                 # events still carries the original bytes
with Model() as model:
    model.run(events, b"AB!")
    model.train(ByteEvents.stream(b"next", link=2, slot=7))
```

`parameters` is a nominal upper bound. The allocator grows the expert banks in complete groups and reduces the default node count for small budgets. Use `experts` to specify capacity directly. Depth, nodes, facets, live nodes, bank count and per-event execution budget can also be set explicitly. For example, `Model(levels=6, nodes=8, facets=3, live_nodes=3, experts=64, active_experts=4)` keeps the full depth in a smaller configuration. `stats()` reports the resolved sizes and actual expert executions by bank.

`configure(fast_rate=..., query_rate=..., route_rate=..., head_rate=...)` controls the MLP, HAT, MFS and readout learning rates, respectively. `clone()` copies a trainable or frozen stream. `inference_copy()` starts frozen with empty context, and `cold=False` retains context. `evaluate()` and `generate()` use cold frozen copies. `reset_context()` clears stream memory and counters. Online updates complete on each target, so `finish()` is an immediate boundary.

## CLI and checkpoints

```bash
shoebox-core train --data input.txt --targets 1000 --out model.shoe
shoebox-core inspect --checkpoint model.shoe
shoebox-core generate --checkpoint model.shoe --prompt 'One ' --length 128
shoebox-core train --resume model.shoe --data input.txt --offset 1000 --out continued.shoe
```

Training prints the next input offset. A `SHOECR02` checkpoint stores weights, the leaf graph, HAT matrices, rates, trace and training status in a SHA-256 integrity envelope. Reloading resumes at the next supplied pair. The caller retains the file cursor.

Version 0.1 checkpoints use `SHOECR01` and describe the former summarizer model. They require the original reader. The v0.2 loader identifies them explicitly. [Version 0.1 source and installation](https://github.com/JRowe47/SHoEbox-core/tree/647e3207d30305cd77c327b48be9a7844795141d)
