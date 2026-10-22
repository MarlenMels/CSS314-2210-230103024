"""Task 7: compare memory-bandwidth and compute scaling limits."""

from __future__ import annotations

import ctypes
import math
import os
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STUDENT_ID = 230103024
ELEMENT_COUNT = 30_000_000
REPETITIONS = 3

NATIVE_SOURCE = r"""
#include <math.h>
#include <stddef.h>
#include <string.h>

void copy_kernel(
    const double *input,
    double *output,
    size_t start,
    size_t stop
) {
    memcpy(output + start, input + start, (stop - start) * sizeof(double));
}

void compute_kernel(
    const double *input,
    double *output,
    size_t start,
    size_t stop
) {
    for (size_t index = start; index < stop; ++index) {
        double value = input[index];
        output[index] = sin(value) * cos(value) + tan(value);
    }
}
"""


@dataclass(frozen=True)
class BenchmarkResult:
    kernel: str
    cores: int
    wall_ms: float
    speedup: float
    efficiency: float


def build_native_helper(directory: Path) -> ctypes.CDLL:
    """Compile optimized copy and trigonometric kernels."""
    source_path = directory / "task7_native.c"
    source_path.write_text(NATIVE_SOURCE, encoding="utf-8")

    if platform.system() == "Darwin":
        library_path = directory / "task7_native.dylib"
        shared_flags = ["-dynamiclib"]
        libraries: list[str] = []
    else:
        library_path = directory / "task7_native.so"
        shared_flags = ["-shared", "-fPIC"]
        libraries = ["-lm"]

    compiler = os.environ.get("CC", "cc")
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c11",
            *shared_flags,
            str(source_path),
            *libraries,
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    library = ctypes.CDLL(str(library_path))
    signature = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    library.copy_kernel.argtypes = signature
    library.copy_kernel.restype = None
    library.compute_kernel.argtypes = signature
    library.compute_kernel.restype = None
    return library


def start_all_workers(executor: ThreadPoolExecutor, worker_count: int) -> None:
    """Create all pool threads before timing any kernel."""
    barrier = threading.Barrier(worker_count + 1)
    futures = [executor.submit(barrier.wait) for _ in range(worker_count)]
    barrier.wait()
    for future in futures:
        future.result()


def run_parallel_kernel(
    executor: ThreadPoolExecutor,
    function: ctypes._CFuncPtr,  # type: ignore[attr-defined]
    input_pointer: ctypes.POINTER(ctypes.c_double),
    output_pointer: ctypes.POINTER(ctypes.c_double),
    element_count: int,
    core_count: int,
) -> None:
    """Split one kernel into equal contiguous ranges."""
    futures = []
    for worker in range(core_count):
        start = worker * element_count // core_count
        stop = (worker + 1) * element_count // core_count
        futures.append(
            executor.submit(
                function,
                input_pointer,
                output_pointer,
                start,
                stop,
            )
        )
    for future in futures:
        future.result()


def benchmark_kernel(
    kernel_name: str,
    function: ctypes._CFuncPtr,  # type: ignore[attr-defined]
    executor: ThreadPoolExecutor,
    source: np.ndarray,
    destination: np.ndarray,
    core_counts: list[int],
) -> list[BenchmarkResult]:
    """Measure one kernel and calculate scaling from its one-core time."""
    input_pointer = source.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_pointer = destination.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    median_times: list[float] = []

    for core_count in core_counts:
        run_parallel_kernel(
            executor,
            function,
            input_pointer,
            output_pointer,
            source.size,
            core_count,
        )

        samples_ns = []
        for _ in range(REPETITIONS):
            start_ns = time.perf_counter_ns()
            run_parallel_kernel(
                executor,
                function,
                input_pointer,
                output_pointer,
                source.size,
                core_count,
            )
            samples_ns.append(time.perf_counter_ns() - start_ns)
        median_times.append(statistics.median(samples_ns) / 1_000_000)

    one_core_ms = median_times[0]
    return [
        BenchmarkResult(
            kernel=kernel_name,
            cores=core_count,
            wall_ms=wall_ms,
            speedup=one_core_ms / wall_ms,
            efficiency=(one_core_ms / wall_ms) / core_count,
        )
        for core_count, wall_ms in zip(core_counts, median_times)
    ]


def validate_results(source: np.ndarray, destination: np.ndarray, compute: bool) -> None:
    """Validate representative values without allocating another large array."""
    for index in (0, source.size // 2, source.size - 1):
        value = source[index]
        expected = (
            math.sin(value) * math.cos(value) + math.tan(value)
            if compute
            else value
        )
        if not math.isclose(destination[index], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Kernel result validation failed at index {index}")


def main() -> None:
    max_cores = os.cpu_count() or 1
    core_counts = list(dict.fromkeys([1, 2, max_cores]))
    source = np.linspace(0.001, 1.0, ELEMENT_COUNT, dtype=np.float64)
    destination = np.empty_like(source)
    destination.fill(0.0)

    with tempfile.TemporaryDirectory(prefix="task7_bandwidth_") as temporary_directory:
        native = build_native_helper(Path(temporary_directory))
        with ThreadPoolExecutor(max_workers=max_cores) as executor:
            start_all_workers(executor, max_cores)
            copy_results = benchmark_kernel(
                "A: Memory copy",
                native.copy_kernel,
                executor,
                source,
                destination,
                core_counts,
            )
            validate_results(source, destination, compute=False)

            compute_results = benchmark_kernel(
                "B: Trigonometric",
                native.compute_kernel,
                executor,
                source,
                destination,
                core_counts,
            )
            validate_results(source, destination, compute=True)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Elements per kernel: {ELEMENT_COUNT}")
    print(f"Maximum logical cores: {max_cores}")
    print(f"Timing statistic: median of {REPETITIONS} runs")
    print()
    print("Kernel             | Cores | Wall Time (ms) | Speedup | Efficiency")
    print("-------------------|-------|----------------|---------|-----------")
    for result in (*copy_results, *compute_results):
        print(
            f"{result.kernel:18s} | {result.cores:5d} | "
            f"{result.wall_ms:14.3f} | {result.speedup:7.3f} | "
            f"{result.efficiency:10.3f}"
        )

    memory_max = copy_results[-1]
    compute_max = compute_results[-1]
    print()
    print(
        "Scaling diagnosis: Kernel A (Memory copy) stalled due to shared RAM "
        "memory-channel saturation."
    )
    print(
        f"At {max_cores} cores its efficiency was {memory_max.efficiency:.3f}, "
        f"versus {compute_max.efficiency:.3f} for compute-bound Kernel B."
    )


if __name__ == "__main__":
    main()
