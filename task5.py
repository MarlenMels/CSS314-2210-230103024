"""Task 5: find where parallel vector work overtakes serial execution."""

from __future__ import annotations

import os

# Keep each NumPy operation single-threaded. Parallelism is controlled only by
# the explicit worker pool below.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"

import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


STUDENT_ID = 230103024
ARRAY_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A = 2.5
B = 1.25
REPETITIONS = 7


@dataclass(frozen=True)
class BenchmarkResult:
    size: int
    serial_ms: float
    parallel_ms: float

    @property
    def speedup(self) -> float:
        return self.serial_ms / self.parallel_ms

    @property
    def winner(self) -> str:
        return "Parallel" if self.parallel_ms < self.serial_ms else "Serial"


def serial_operation(x: np.ndarray, y: np.ndarray) -> None:
    """Calculate y = A*x + B using one calling thread and no allocations."""
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def vector_chunk(x: np.ndarray, y: np.ndarray, start: int, stop: int) -> None:
    """Calculate one non-overlapping output slice."""
    x_chunk = x[start:stop]
    y_chunk = y[start:stop]
    np.multiply(x_chunk, A, out=y_chunk)
    np.add(y_chunk, B, out=y_chunk)


def parallel_operation(
    executor: ThreadPoolExecutor,
    x: np.ndarray,
    y: np.ndarray,
    worker_count: int,
) -> None:
    """Calculate y = A*x + B across all available workers."""
    chunk_size = (x.size + worker_count - 1) // worker_count
    futures = [
        executor.submit(vector_chunk, x, y, start, min(start + chunk_size, x.size))
        for start in range(0, x.size, chunk_size)
    ]
    for future in futures:
        future.result()


def elapsed_ns(operation: object, *arguments: object) -> int:
    """Measure one completed operation with a monotonic high-resolution clock."""
    start_ns = time.perf_counter_ns()
    operation(*arguments)  # type: ignore[operator]
    return time.perf_counter_ns() - start_ns


def start_all_workers(executor: ThreadPoolExecutor, worker_count: int) -> None:
    """Create every pool thread before benchmark timing begins."""
    barrier = threading.Barrier(worker_count + 1)
    futures = [executor.submit(barrier.wait) for _ in range(worker_count)]
    barrier.wait()
    for future in futures:
        future.result()


def benchmark_size(
    size: int,
    executor: ThreadPoolExecutor,
    worker_count: int,
) -> BenchmarkResult:
    """Return median serial and parallel times using alternating run order."""
    rng = np.random.default_rng(STUDENT_ID + size)
    x = rng.random(size, dtype=np.float64)
    serial_y = np.empty_like(x)
    parallel_y = np.empty_like(x)

    # Warm both paths before recording timings.
    serial_operation(x, serial_y)
    parallel_operation(executor, x, parallel_y, worker_count)

    serial_times: list[int] = []
    parallel_times: list[int] = []
    for repetition in range(REPETITIONS):
        if repetition % 2 == 0:
            serial_times.append(elapsed_ns(serial_operation, x, serial_y))
            parallel_times.append(
                elapsed_ns(parallel_operation, executor, x, parallel_y, worker_count)
            )
        else:
            parallel_times.append(
                elapsed_ns(parallel_operation, executor, x, parallel_y, worker_count)
            )
            serial_times.append(elapsed_ns(serial_operation, x, serial_y))

    expected_first = A * x[0] + B
    expected_last = A * x[-1] + B
    for output in (serial_y, parallel_y):
        if not (
            math.isclose(output[0], expected_first, rel_tol=1e-12)
            and math.isclose(output[-1], expected_last, rel_tol=1e-12)
        ):
            raise RuntimeError(f"Vector result validation failed for N={size}")

    return BenchmarkResult(
        size=size,
        serial_ms=statistics.median(serial_times) / 1_000_000,
        parallel_ms=statistics.median(parallel_times) / 1_000_000,
    )


def crossover_ranges(results: list[BenchmarkResult]) -> list[tuple[int, int]]:
    """Find tested-size brackets that change from serial to parallel winner."""
    ranges = []
    for smaller, larger in zip(results, results[1:]):
        if smaller.winner == "Serial" and larger.winner == "Parallel":
            ranges.append((smaller.size, larger.size))
    return ranges


def main() -> None:
    worker_count = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        start_all_workers(executor, worker_count)
        results = [
            benchmark_size(size, executor, worker_count) for size in ARRAY_SIZES
        ]

    print(f"Student ID: {STUDENT_ID}")
    print(f"Available logical cores / workers: {worker_count}")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print(f"Timing statistic: median of {REPETITIONS} runs")
    print()
    print("N          | Serial (ms) | Parallel (ms) | Speedup | Faster")
    print("-----------|-------------|---------------|---------|---------")
    for result in results:
        print(
            f"{result.size:10d} | {result.serial_ms:11.3f} | "
            f"{result.parallel_ms:13.3f} | {result.speedup:7.3f} | "
            f"{result.winner}"
        )

    ranges = crossover_ranges(results)
    print()
    if ranges:
        for lower, upper in ranges:
            print(
                "Measured slower-to-faster crossover range: "
                f"{lower:,} < N <= {upper:,}"
            )
    elif all(result.winner == "Parallel" for result in results):
        print(
            "Parallel was already faster at the smallest tested size; "
            f"crossover is at N <= {ARRAY_SIZES[0]:,}."
        )
    else:
        print("No slower-to-faster crossover occurred within the tested N range.")


if __name__ == "__main__":
    main()
