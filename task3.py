"""Task 3: measure cache-line invalidation caused by false sharing.

The tight loops run in a temporary native helper because CPython's GIL would
otherwise serialize Python bytecode and hide the hardware effect being tested.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path


STUDENT_ID = 230103024
INCREMENTS_PER_THREAD = 50_000_000

NATIVE_SOURCE = r"""
#include <stddef.h>
#include <stdint.h>

/* Adjacent uint64_t elements share cache lines and invalidate one another. */
void increment_shared(uint64_t *shared_counts, size_t index, uint64_t iterations) {
    volatile uint64_t *counter = &shared_counts[index];
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
        (*counter)++;
    }
}

/* The hot counter stays on the worker's stack; shared memory is written once. */
void increment_local(uint64_t *shared_counts, size_t index, uint64_t iterations) {
    volatile uint64_t local_count = 0;
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
        local_count++;
    }
    shared_counts[index] = local_count;
}
"""


def physical_core_count() -> int:
    """Return physical cores where the host OS exposes that information."""
    if platform.system() == "Darwin":
        value = ctypes.c_uint32()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        libc = ctypes.CDLL(None)
        result = libc.sysctlbyname(
            b"hw.physicalcpu",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        )
        if result == 0 and value.value > 0:
            return value.value

    if platform.system() == "Linux":
        try:
            records = Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n")
            physical_cores: set[tuple[str, str]] = set()
            for record in records:
                fields = {}
                for line in record.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip()] = value.strip()
                if "physical id" in fields and "core id" in fields:
                    physical_cores.add((fields["physical id"], fields["core id"]))
            if physical_cores:
                return len(physical_cores)
        except OSError:
            pass

    return os.cpu_count() or 1


def build_native_helper(directory: Path) -> ctypes.CDLL:
    """Compile the increment loops into a temporary shared library."""
    source_path = directory / "task3_native.c"
    source_path.write_text(NATIVE_SOURCE, encoding="utf-8")

    if platform.system() == "Darwin":
        library_path = directory / "task3_native.dylib"
        shared_flags = ["-dynamiclib"]
    else:
        library_path = directory / "task3_native.so"
        shared_flags = ["-shared", "-fPIC"]

    compiler = os.environ.get("CC", "cc")
    command = [
        compiler,
        "-O2",
        "-std=c11",
        *shared_flags,
        str(source_path),
        "-o",
        str(library_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    library = ctypes.CDLL(str(library_path))
    signature = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    library.increment_shared.argtypes = signature
    library.increment_shared.restype = None
    library.increment_local.argtypes = signature
    library.increment_local.restype = None
    return library


def benchmark(
    function: ctypes._CFuncPtr,  # type: ignore[attr-defined]
    shared_counts: ctypes.Array[ctypes.c_uint64],
    worker_count: int,
) -> float:
    """Run one native increment function on synchronized Python threads."""
    for index in range(worker_count):
        shared_counts[index] = 0

    start_barrier = threading.Barrier(worker_count + 1)

    def worker(index: int) -> None:
        start_barrier.wait()
        # Functions loaded through ctypes.CDLL release the GIL for this call.
        function(shared_counts, index, INCREMENTS_PER_THREAD)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()

    start_ns = time.perf_counter_ns()
    start_barrier.wait()
    for thread in threads:
        thread.join()
    elapsed_ns = time.perf_counter_ns() - start_ns

    expected_total = worker_count * INCREMENTS_PER_THREAD
    actual_total = sum(shared_counts)
    if actual_total != expected_total:
        raise RuntimeError(f"Counter validation failed: {actual_total} != {expected_total}")

    return elapsed_ns / 1_000_000


def main() -> None:
    worker_count = physical_core_count()
    shared_counts_type = ctypes.c_uint64 * worker_count
    shared_counts = shared_counts_type()

    with tempfile.TemporaryDirectory(prefix="task3_false_sharing_") as temporary_directory:
        native = build_native_helper(Path(temporary_directory))
        unpadded_ms = benchmark(native.increment_shared, shared_counts, worker_count)
        local_ms = benchmark(native.increment_local, shared_counts, worker_count)

    ratio = unpadded_ms / local_ms
    expected_total = worker_count * INCREMENTS_PER_THREAD

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores / worker threads: {worker_count}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD}")
    print(f"Validated total per run: {expected_total}")
    print(f"Unpadded Shared: {unpadded_ms:.3f} ms")
    print(f"Padded/Local: {local_ms:.3f} ms")
    print(f"Execution ratio (Unpadded / Padded): {ratio:.3f}x")


if __name__ == "__main__":
    main()
