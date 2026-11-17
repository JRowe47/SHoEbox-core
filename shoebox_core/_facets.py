"""Native FP32 multifacet softmax and its exact vector-Jacobian product."""
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


def _library():
    global _LIB
    with _BUILD_LOCK:
        if _LIB is not None:
            return _LIB
        compiler = os.environ.get("CXX", "g++")
        flags = ["-std=c++17", "-O3", "-march=native", "-ffp-contract=off", "-fPIC", "-shared"]
        source = (HERE / "native.cpp").read_bytes()
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
        array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        for name, buffers in (("facet_forward", 5), ("facet_backward", 8)):
            fn = getattr(lib, name)
            fn.argtypes = [C.c_int] * 4 + [array] * buffers
            fn.restype = None
        _LIB = lib
        return lib


def forward(queries, keys, mixture):
    parents, facets, _ = queries.shape
    children = keys.shape[-2]
    stride = 0 if keys.ndim == 2 else children * keys.shape[-1]
    sizes = (parents, facets, children, stride)
    probabilities = np.empty((parents, facets, children), np.float32)
    output = np.empty((parents, children), np.float32)
    _library().facet_forward(*sizes, queries, keys, mixture, probabilities, output)
    return output, (sizes, queries, keys, mixture, probabilities)


def backward(tape, incoming):
    sizes, queries, keys, mixture, probabilities = tape
    dq, dk = np.zeros_like(queries), np.zeros_like(keys)
    dmix = np.empty_like(mixture)
    _library().facet_backward(*sizes, queries, keys, mixture, probabilities, incoming,
                              dq, dk, dmix)
    return dq, dk, dmix
