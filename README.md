# SHoEbox-core

The standalone SHoEbox byte model and online learning algorithm. A small
Python API drives a C++17 CPU runtime with NumPy as its only Python dependency.

## Install

Requires Python 3.10+, NumPy 1.26+, and a C++17 compiler on Linux or WSL.

```bash
git clone https://github.com/JRowe47/SHoEbox-core.git
cd SHoEbox-core
python -m pip install .
```

The first model compiles the native runtime into a temporary cache. `CXX`
selects the compiler; `SHOEBOX_CORE_CACHE` selects the cache directory.
Floating-point contraction is disabled to preserve the extracted arithmetic.

## Use

```python
from shoebox_core import Model

with Model(parameters=1_000_000, seed=7) as model:
    loss = model.train(b"An observed training sequence.\n")
    model.save("model.shoe")
    print(model.parameter_count)
    print(model.evaluate(b"A separate sequence.\n"))
    print(model.generate("An ", length=128).decode("utf-8", "replace"))

with Model.load("model.shoe") as resumed:
    resumed.train(b"The next training sequence.\n")
```

`train(data)` consumes `len(data)-1` next-byte pairs and returns mean negative
log likelihood in nats per byte. Text is encoded as UTF-8. Bytes and
one-dimensional NumPy `uint8` arrays are accepted directly.

For a continuous stream, preserve the cross-chunk next-byte pair explicitly:

```python
data = b"A continuous byte stream with no dropped boundary pairs."
with Model() as model:
    for start in range(0, len(data)-1, 16):
        end = min(start+16, len(data)-1)
        model.run(data[start:end], data[start+1:end+1])
    model.save("stream.shoe")
```

```bash
shoebox-core train --data input.txt --targets 1000 --out model.shoe
shoebox-core inspect --checkpoint model.shoe
shoebox-core generate --checkpoint model.shoe --prompt 'An ' --length 128
shoebox-core train --resume model.shoe --data input.txt --offset 1000 --out continued.shoe
```

The training command prints `next_offset`, the input index to supply when
continuing the same file. The input file and its cursor belong to the caller;
the checkpoint contains model state. `python -m shoebox_core` exposes the same CLI.

## Model

Inputs and outputs cover all 256 bytes. State width is 64. Four clocks compose
groups of four ordered children:

| Scale | Bytes per event | Computation |
| --- | ---: | --- |
| Byte | 1 | One selected sparse expert, recurrent transition |
| Group | 4 | Ordered composition, one selected sparse expert |
| Summary | 16 | Ordered composition, diagonal recurrence |
| Summary | 64 | Ordered composition, diagonal recurrence |

The two expert banks share the remaining parameter budget in approximately a
2:1 ratio. Each event uses centered, normalized content to score every expert
prototype exactly. The winning expert executes; the runner-up supplies a
margin gate and its derivative. Initial prototype seeding uses observed
training cues. A coverage bias fades as the least-used expert approaches
256 selections, then accumulated bias decays.

Every scale carries a compact recurrent matrix memory. The slower scales also
have content-and-time addressed memory with residual writes and fixed rotating
clocks. Each scale retains up to 16 ordered constituent pairs. A learned query
reads these retained values on every byte, including between slow events.
Stored values remain detached after publication.

Upper summaries contribute learned constant gain/bias modulation; the
4-byte level supplies state-dependent modulation. A binary-tree decoder with
255 internal nodes defines normalized probabilities for all 256 next bytes.
Training visits the eight decisions for the observed target; `logits()`
returns all 256 normalized log probabilities.

## Learning

Learning is online and local, using explicit derivatives and clipped SGD:

1. Consume an observed input, update the clocks and memories, and predict the
   next byte. The target is supplied only after observation.
2. Update the byte decoder and retained-memory query/read heads per target.
   Accumulate derivatives for the recurrent states and output modulation.
3. Retire recurrent and memory derivatives in blocks of eight events at each
   scale. The entering recurrent state and matrix memories are detached at
   those local boundaries.
4. Keep a bounded tree of content derivatives so completed parents can credit
   their ordered children. Retire it from the top scale downward every
   `8 * 4**(levels-1)` input bytes: 512 bytes for the four-scale model.
5. Apply the 4-byte contrastive next-constituent objective and slower-scale
   child reconstruction objective alongside next-byte learning. Update expert
   weights, prototype attraction, ordered composition and byte embeddings at
   their corresponding local boundaries.

Default rates are fast/query `0.008`, slow `0.08`, head `0.16`, and prototype
attraction `0.002`. Auxiliary and reconstruction weights are `0.1` and `0.05`.
`model.configure(...)` changes these independently without resetting state.
There is no autograd framework or optimizer-moment state.

## State and checkpoints

`save()` preserves unfinished groups, pending gradients, routing statistics,
memory and trainable/frozen status. It does not apply pending updates.
`finish()` explicitly retires pending credit. `reset_context()` first finishes
credit, then clears recurrent context and retained memory while preserving
weights and learned routing statistics.

`clone()` copies the complete model. `inference_copy()` freezes current weights
and starts with empty context; use `cold=False` to retain current context.
`evaluate()` and `generate()` operate on frozen copies and leave the source
unchanged. `observe(byte)` is available on frozen copies.

The standalone `shoebox-core-1` checkpoint has a SHA-256 integrity envelope
and native layout/endian checks. It is separate from upstream checkpoint
formats; load existing SHoEbox checkpoints with their original repository.
Use separate model instances for concurrent streams.

## Source

Extracted from the general-purpose minimal core (`v19`) in
[JRowe47/SHoEbox at 696cc8d](https://github.com/JRowe47/SHoEbox/tree/696cc8d9f7e46638251a67feef0ceb15a1ef611f).
The numerical core comes from `multiscale/native.cpp`, `native.hpp`,
`address_memory.hpp`, `recall.hpp`, and the centered routing implementation.

This repository contains only the model, learning code, Python API, CLI and
installation metadata. Task-specific E8 extraction heads, experimental
extensions, historical backends, external key/value storage, development
harnesses, reports, datasets and trained weights are excluded.

| File | Purpose |
| --- | --- |
| `shoebox_core/native.cpp` | Forward pass, learning, checkpoints and C ABI |
| `shoebox_core/native.hpp` | Model state and vector primitives |
| `shoebox_core/address_memory.hpp` | Addressed recurrent memory and reconstruction |
| `shoebox_core/recall.hpp` | Bounded constituent retention and query learning |
| `shoebox_core/routing*.hpp` | Exact prototype routing and coverage warmup |
| `shoebox_core/model.py` | Python lifecycle, byte API and compiler cache |
| `shoebox_core/__main__.py` | Local-file training, inspection and generation |
