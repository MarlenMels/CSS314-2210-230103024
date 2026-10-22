"""Task 6: expose lost updates and compare them with atomic increments.

The increment loops run in a temporary native helper because CPython's GIL
would otherwise serialize Python bytecode and conceal the race condition.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path


STUDENT_ID = 230103024
THREAD_COUNT = 4
INCREMENTS_PER_THREAD = 2_000_000

NATIVE_SOURCE = r"""
#include <stdint.h>

/*
 * Each load and store is atomic, but the overall read-modify-write is not.
 * Threads can therefore overwrite one another's increments.
 */
void increment_racy(uint64_t *counter, uint64_t iterations) {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
        uint64_t observed = __atomic_load_n(counter, __ATOMIC_RELAXED);
        __atomic_store_n(counter, observed + 1, __ATOMIC_RELAXED);
    }
}

/* One indivisible hardware-supported read-modify-write per increment. */
void increment_atomic(uint64_t *counter, uint64_t iterations) {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
        __atomic_fetch_add(counter, 1, __ATOMIC_RELAXED);
    }
}
"""


@dataclass(frozen=True)
class BenchmarkResult:
    final_value: int
    elapsed_ms: float


def build_native_helper(directory: Path) -> ctypes.CDLL:
    """Compile both counter loops into a temporary shared library."""
    source_path = directory / "task6_native.c"
    source_path.write_text(NATIVE_SOURCE, encoding="utf-8")

    if platform.system() == "Darwin":
        library_path = directory / "task6_native.dylib"
        shared_flags = ["-dynamiclib"]
    else:
        library_path = directory / "task6_native.so"
        shared_flags = ["-shared", "-fPIC"]

    compiler = os.environ.get("CC", "cc")
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c11",
            *shared_flags,
            str(source_path),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    library = ctypes.CDLL(str(library_path))
    signature = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
    library.increment_racy.argtypes = signature
    library.increment_racy.restype = None
    library.increment_atomic.argtypes = signature
    library.increment_atomic.restype = None
    return library


def benchmark(function: ctypes._CFuncPtr) -> BenchmarkResult:  # type: ignore[attr-defined]
    """Call one counter implementation concurrently from four threads."""
    shared_counter = ctypes.c_uint64(0)
    start_barrier = threading.Barrier(THREAD_COUNT + 1)

    def worker() -> None:
        start_barrier.wait()
        # ctypes.CDLL releases the GIL while the native function is executing.
        function(ctypes.byref(shared_counter), INCREMENTS_PER_THREAD)

    threads = [threading.Thread(target=worker) for _ in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()

    start_ns = time.perf_counter_ns()
    start_barrier.wait()
    for thread in threads:
        thread.join()
    elapsed_ns = time.perf_counter_ns() - start_ns

    return BenchmarkResult(
        final_value=shared_counter.value,
        elapsed_ms=elapsed_ns / 1_000_000,
    )


def main() -> None:
    expected = THREAD_COUNT * INCREMENTS_PER_THREAD

    with tempfile.TemporaryDirectory(prefix="task6_race_") as temporary_directory:
        native = build_native_helper(Path(temporary_directory))
        unsafe = benchmark(native.increment_racy)
        atomic = benchmark(native.increment_atomic)

    if unsafe.final_value > expected:
        raise RuntimeError("Unsafe counter unexpectedly exceeded the target")
    if atomic.final_value != expected:
        raise RuntimeError("Atomic counter validation failed")

    updates_lost = expected - unsafe.final_value
    lost_percentage = updates_lost / expected * 100
    penalty_ms = atomic.elapsed_ms - unsafe.elapsed_ms
    penalty_ratio = atomic.elapsed_ms / unsafe.elapsed_ms

    print(f"Student ID: {STUDENT_ID}")
    print(f"Threads: {THREAD_COUNT}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD}")
    print(f"Expected final value: {expected}")
    print(f"Actual final value (unsynchronized): {unsafe.final_value}")
    print(f"Updates lost: {updates_lost} ({lost_percentage:.3f}%)")
    print(f"Unsynchronized wall time: {unsafe.elapsed_ms:.3f} ms")
    print(f"Atomic final value: {atomic.final_value}")
    print(f"Atomic wall time: {atomic.elapsed_ms:.3f} ms")
    print(
        f"Atomic synchronization penalty: {penalty_ms:.3f} ms "
        f"({penalty_ratio:.3f}x slower)"
    )


if __name__ == "__main__":
    main()
