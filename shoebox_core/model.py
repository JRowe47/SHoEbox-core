"""NumPy interface to the self-contained C++17 model and online learner."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import threading

import numpy as np

HERE = Path(__file__).resolve().parent
_LIB = None
_BUILD_LOCK = threading.Lock()
_RATE_NAMES = ("fast_rate", "slow_rate", "query_rate", "head_rate",
               "auxiliary_weight", "route_rate", "reconstruction_weight")


def _library():
    global _LIB
    with _BUILD_LOCK:
        if _LIB is not None:
            return _LIB
        compiler = os.environ.get("CXX", "g++")
        flags = ["-std=c++17", "-O3", "-march=native", "-ffp-contract=off", "-fPIC", "-shared"]
        source = b"".join(p.name.encode() + p.read_bytes() for p in sorted(HERE.glob("*.hpp")))
        source += (HERE / "native.cpp").read_bytes()
        signature = subprocess.check_output([compiler, "--version"])
        # Native instructions are host-specific; include the CPU fingerprint
        # when a cache directory is shared between machines.
        cpu = Path("/proc/cpuinfo")
        hardware = platform.machine().encode()
        if cpu.exists():
            hardware += "\n".join(sorted({line for line in cpu.read_text().splitlines()
                                         if line.startswith(("vendor_id", "model name", "flags", "Features"))})).encode()
        key = hashlib.sha256(source + repr(flags).encode() + signature + hardware).hexdigest()[:24]
        cache = Path(os.environ.get("SHOEBOX_CORE_CACHE", str(Path(tempfile.gettempdir()) / "shoebox_core")))
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"core_{key}.so"
        if not path.exists():
            fd, name = tempfile.mkstemp(prefix="build_", suffix=".so", dir=cache)
            os.close(fd)
            temporary = Path(name)
            try:
                subprocess.run([compiler, *flags, str(HERE / "native.cpp"), "-o", name], check=True)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        lib = C.CDLL(str(path))
        p, i, n = C.c_void_p, C.c_int, C.c_size_t
        specs = {
            "ms_error": ([], C.c_char_p),
            "ms_create": ([i, i, C.c_uint32], p),
            "ms_destroy": ([p], None), "ms_clone": ([p, i, i], p),
            "ms_run": ([p, p, p, n, C.POINTER(C.c_double)], i),
            "ms_observe": ([p, C.c_uint8, p], i), "ms_logits": ([p, p], i),
            "ms_reset": ([p], i), "ms_finish": ([p], i),
            "ms_configure": ([p, p], i), "ms_stats": ([p, p], i),
            "ms_save_size": ([p], n), "ms_save": ([p, p, n], i), "ms_load": ([p, n], p),
        }
        for name, (args, result) in specs.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, result
        _LIB = lib
        return lib


def _bytes(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    array = np.asarray(data)
    if array.ndim != 1 or array.dtype != np.uint8:
        raise TypeError("expected bytes, text, or a one-dimensional uint8 array")
    return np.ascontiguousarray(array)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


class Model:
    """Sparse byte model with 1/4-byte experts and 16/64-byte summaries.

    `parameters` is a nominal fitted-parameter budget; `parameter_count` gives
    the actual count. `levels` keeps the first 1..4 scales. Width remains 64.
    Each instance owns mutable stream state; use separate clones for concurrent
    work. Creating an inference copy never changes the source's pending credit.
    """

    def __init__(self, *, parameters=1_000_000, seed=7, levels=4):
        parameters = _integer(parameters, "parameters", 50_000, 100_000_000)
        seed = _integer(seed, "seed", 0, 2**32 - 1)
        levels = _integer(levels, "levels", 1, 4)
        self._lib = _library()
        self._handle = self._lib.ms_create(levels, parameters, seed)
        self._check_pointer()

    def _check_pointer(self):
        if not self._handle:
            raise RuntimeError(self._lib.ms_error().decode())

    def _check(self, code):
        if code:
            raise RuntimeError(self._lib.ms_error().decode())

    def _alive(self):
        if not self._handle:
            raise RuntimeError("model has been closed")
        return self._handle

    @classmethod
    def _wrap(cls, handle):
        obj = cls.__new__(cls)
        obj._lib, obj._handle = _library(), handle
        obj._check_pointer()
        return obj

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.ms_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        self._alive()
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, inputs, targets):
        """Consume aligned input/target bytes and return mean NLL in nats/byte.

        Trainable instances update online. Frozen copies only score. Call with
        consecutive x/y slices to retain every next-byte pair across chunks.
        """
        x, y = _bytes(inputs), _bytes(targets)
        if x.size != y.size or not x.size:
            raise ValueError("inputs and targets must have the same nonzero length")
        loss = C.c_double()
        self._check(self._lib.ms_run(self._alive(), x.ctypes.data, y.ctypes.data, x.size, C.byref(loss)))
        return loss.value

    def train(self, data):
        """Train on len(data)-1 next-byte pairs without resetting or flushing."""
        if not self.stats()["training"]:
            raise RuntimeError("this is a frozen inference model")
        data = _bytes(data)
        return self.run(data[:-1], data[1:])

    def logits(self):
        """Return normalized log probabilities for all 256 next bytes."""
        out = np.empty(256, np.float32)
        self._check(self._lib.ms_logits(self._alive(), out.ctypes.data))
        return out

    def observe(self, byte):
        """Consume one observed byte on a frozen copy; return next-byte log probabilities."""
        byte = _integer(byte, "byte", 0, 255)
        out = np.empty(256, np.float32)
        self._check(self._lib.ms_observe(self._alive(), byte, out.ctypes.data))
        return out

    def clone(self):
        """Copy weights, stream state, pending updates and frozen/trainable status."""
        return self._wrap(self._lib.ms_clone(self._alive(), 0, 0))

    def inference_copy(self, *, cold=True):
        """Freeze current weights; clear context by default, retain it with cold=False."""
        return self._wrap(self._lib.ms_clone(self._alive(), 1, int(bool(cold))))

    def evaluate(self, data):
        """Score from empty context without modifying this model."""
        data = _bytes(data)
        with self.inference_copy() as model:
            return model.run(data[:-1], data[1:])

    def generate(self, prompt, *, length=256, temperature=.8, seed=7):
        """Return the UTF-8 prompt bytes plus sampled continuation bytes."""
        data = _bytes(prompt)
        length = _integer(length, "length", 0, 2**63 - 1)
        if not data.size or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("a nonempty prompt and finite positive temperature are required")
        rng = np.random.default_rng(seed)
        result = bytearray(data.tobytes())
        with self.inference_copy() as model:
            for byte in data:
                logits = model.observe(int(byte))
            for _ in range(length):
                z = logits.astype(np.float64) / temperature
                probabilities = np.exp(z - z.max())
                byte = int(rng.choice(256, p=probabilities / probabilities.sum()))
                result.append(byte)
                logits = model.observe(byte)
        return bytes(result)

    def finish(self):
        """Apply pending local gradients at an explicit truncation boundary."""
        self._check(self._lib.ms_finish(self._alive()))

    def reset_context(self):
        """Finish pending credit, then clear recurrent state and retained memory."""
        self._check(self._lib.ms_reset(self._alive()))

    def configure(self, *, fast_rate=None, slow_rate=None, query_rate=None,
                  head_rate=None, auxiliary_weight=None, route_rate=None,
                  reconstruction_weight=None):
        """Set independent learning rates/weights; omitted values stay unchanged."""
        updates = (fast_rate, slow_rate, query_rate, head_rate, auxiliary_weight,
                   route_rate, reconstruction_weight)
        current = self.stats()["learning_rates"]
        values = [current[name] if value is None else value for name, value in zip(_RATE_NAMES, updates)]
        if any(not np.isfinite(v) or v < 0 or v > np.finfo(np.float32).max for v in values):
            raise ValueError("rates and weights must be finite nonnegative float32 values")
        rates = np.asarray(values, dtype=np.float32)
        self._check(self._lib.ms_configure(self._alive(), rates.ctypes.data))

    def stats(self):
        out = np.empty(80, np.float64)
        self._check(self._lib.ms_stats(self._alive(), out.ctypes.data))
        count = int(out[3])
        names = ("experts", "events", "updates", "pending", "constituent_pending",
                 "pending_children", "retained_units", "native_targets", "native_loss_sum",
                 "reconstruction_targets", "reconstruction_loss_sum", "state_energy",
                 "routing_events", "routing_seeded", "constituent_credit_edges", "content_updates")
        result = dict(parameters=int(out[0]), bytes=int(out[1]), training=bool(out[2]),
                      levels=count, level_kinds=["expert" if i < 2 else "summary" for i in range(count)],
                      constituent_epoch_bytes=int(out[4]), constituent_epoch_limit=int(out[5]),
                      route_comparisons=int(out[6]), checkpoint_version="shoebox-core-1",
                      learning_rates=dict(zip(_RATE_NAMES, out[8:15].tolist())))
        for j, name in enumerate(names):
            cast = float if j in (8, 10, 11) else int
            result[name] = [cast(out[16 + 16*i + j]) for i in range(count)]
        return result

    @property
    def parameter_count(self):
        return self.stats()["parameters"]

    def dumps(self):
        """Serialize exact state without applying pending updates."""
        size = self._lib.ms_save_size(self._alive())
        if not size:
            raise RuntimeError(self._lib.ms_error().decode())
        buffer = C.create_string_buffer(size)
        self._check(self._lib.ms_save(self._alive(), buffer, size))
        raw = buffer.raw
        return b"SHOECR01" + hashlib.sha256(raw).digest() + raw

    @classmethod
    def loads(cls, data):
        data = bytes(data)
        if len(data) < 40 or data[:8] != b"SHOECR01" or hashlib.sha256(data[40:]).digest() != data[8:40]:
            raise ValueError("invalid or corrupted SHoEbox-core checkpoint")
        raw = data[40:]
        return cls._wrap(_library().ms_load(raw, len(raw)))

    def save(self, path):
        """Atomically save a core checkpoint, including unfinished groups."""
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
