"""Task 2: benchmark parallel array summation and Amdahl scaling."""

import os

# Keep each NumPy reduction single-threaded so that the requested worker count
# controls the amount of parallelism.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"

import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


STUDENT_ID = 230103024
ARRAY_SIZE = 20_000_000
PARALLEL_FRACTION = 0.95


def sum_chunk(chunk: np.ndarray) -> np.float64:
    """Sum one non-overlapping array chunk on a worker thread."""
    return np.sum(chunk, dtype=np.float64)


def parallel_sum(values: np.ndarray, worker_count: int) -> tuple[float, int]:
    """Return the chunked sum and elapsed wall-clock time in nanoseconds."""
    chunk_size = (values.size + worker_count - 1) // worker_count
    chunks = [
        values[start : min(start + chunk_size, values.size)]
        for start in range(0, values.size, chunk_size)
    ]

    start_ns = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        partial_sums = list(executor.map(sum_chunk, chunks))
    total = float(np.sum(partial_sums, dtype=np.float64))
    elapsed_ns = time.perf_counter_ns() - start_ns

    return total, elapsed_ns


def amdahl_upper_bound(worker_count: int) -> float:
    """Calculate Amdahl's theoretical speedup for the worker count."""
    serial_fraction = 1.0 - PARALLEL_FRACTION
    return 1.0 / (serial_fraction + PARALLEL_FRACTION / worker_count)


def main() -> None:
    max_cores = os.cpu_count() or 1
    worker_counts = [1, 2, 4, max_cores, max_cores * 2]

    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Logical cores: {max_cores}")
    print(f"Array elements: {ARRAY_SIZE}")
    print(f"Parallel fraction for Amdahl's Law: {PARALLEL_FRACTION:.2f}")

    rng = np.random.default_rng(STUDENT_ID)
    values = rng.random(ARRAY_SIZE, dtype=np.float64)

    # Warm up NumPy before measuring so first-use setup is not charged only to
    # the one-worker baseline.
    np.sum(values[:1024], dtype=np.float64)

    results: list[tuple[int, float, int]] = []
    for worker_count in worker_counts:
        total, elapsed_ns = parallel_sum(values, worker_count)
        results.append((worker_count, total, elapsed_ns))

    baseline_ns = results[0][2]
    expected_sum = results[0][1]
    if not all(np.isclose(total, expected_sum, rtol=1e-12) for _, total, _ in results):
        raise RuntimeError("Parallel sums did not agree with the baseline sum")

    print(f"Validated sum: {expected_sum:.17g}")
    print()
    print("Cores | Wall Time (ms) | Speedup | Amdahl Upper Bound")
    print("------|----------------|---------|-------------------")
    for worker_count, _, elapsed_ns in results:
        elapsed_ms = elapsed_ns / 1_000_000
        speedup = baseline_ns / elapsed_ns
        upper_bound = amdahl_upper_bound(worker_count)
        print(
            f"{worker_count:5d} | {elapsed_ms:14.3f} | "
            f"{speedup:7.3f} | {upper_bound:18.3f}"
        )


if __name__ == "__main__":
    main()
