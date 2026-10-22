"""Task 4: compare static and dynamic scheduling under irregular work."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable


STUDENT_ID = 230103024
ITEM_COUNT = 1_000
WORKER_COUNT = 4
DYNAMIC_CHUNK_SIZE = 5
SLEEP_SCALE_SECONDS = 0.0001


@dataclass(frozen=True)
class BenchmarkResult:
    strategy: str
    total_ms: float
    first_finish_ms: float
    last_finish_ms: float
    finish_delta_ms: float


def process_item(index: int) -> None:
    """Simulate the deterministic, irregular cost for one item."""
    delay = SLEEP_SCALE_SECONDS * (index % 20)
    if delay > 0:
        time.sleep(delay)


def run_workers(
    strategy_name: str,
    worker_body: Callable[[int], int],
) -> BenchmarkResult:
    """Run synchronized workers and collect wall-clock completion times."""
    start_barrier = threading.Barrier(WORKER_COUNT + 1)
    finish_times_ns = [0] * WORKER_COUNT
    processed_counts = [0] * WORKER_COUNT

    def worker(worker_id: int) -> None:
        start_barrier.wait()
        processed_counts[worker_id] = worker_body(worker_id)
        finish_times_ns[worker_id] = time.perf_counter_ns()

    threads = [
        threading.Thread(target=worker, args=(worker_id,))
        for worker_id in range(WORKER_COUNT)
    ]
    for thread in threads:
        thread.start()

    start_ns = time.perf_counter_ns()
    start_barrier.wait()
    for thread in threads:
        thread.join()
    end_ns = time.perf_counter_ns()

    if sum(processed_counts) != ITEM_COUNT:
        raise RuntimeError(
            f"Processed {sum(processed_counts)} items instead of {ITEM_COUNT}"
        )

    first_finish_ns = min(finish_times_ns)
    last_finish_ns = max(finish_times_ns)
    return BenchmarkResult(
        strategy=strategy_name,
        total_ms=(end_ns - start_ns) / 1_000_000,
        first_finish_ms=(first_finish_ns - start_ns) / 1_000_000,
        last_finish_ms=(last_finish_ns - start_ns) / 1_000_000,
        finish_delta_ms=(last_finish_ns - first_finish_ns) / 1_000_000,
    )


def benchmark_static() -> BenchmarkResult:
    """Assign one fixed, equal-sized contiguous block to each worker."""
    block_size = ITEM_COUNT // WORKER_COUNT

    def process_static_block(worker_id: int) -> int:
        start = worker_id * block_size
        stop = ITEM_COUNT if worker_id == WORKER_COUNT - 1 else start + block_size
        for index in range(start, stop):
            process_item(index)
        return stop - start

    return run_workers("Static (250-item blocks)", process_static_block)


def benchmark_dynamic() -> BenchmarkResult:
    """Let workers dynamically claim five-item chunks from a shared queue."""
    work_queue: queue.Queue[tuple[int, int]] = queue.Queue()
    for start in range(0, ITEM_COUNT, DYNAMIC_CHUNK_SIZE):
        work_queue.put((start, min(start + DYNAMIC_CHUNK_SIZE, ITEM_COUNT)))

    def process_dynamic_chunks(_worker_id: int) -> int:
        processed = 0
        while True:
            try:
                start, stop = work_queue.get_nowait()
            except queue.Empty:
                break
            for index in range(start, stop):
                process_item(index)
            processed += stop - start
            work_queue.task_done()
        return processed

    result = run_workers("Dynamic (chunk size 5)", process_dynamic_chunks)
    if not work_queue.empty():
        raise RuntimeError("Dynamic queue was not fully consumed")
    return result


def main() -> None:
    static_result = benchmark_static()
    dynamic_result = benchmark_dynamic()

    print(f"Student ID: {STUDENT_ID}")
    print(f"Items: {ITEM_COUNT}")
    print(f"Workers: {WORKER_COUNT}")
    print("Work per item: sleep(0.0001 * (i % 20))")
    print()
    print(
        "Strategy                 | Total Time (ms) | First Finish (ms) | "
        "Last Finish (ms) | Finish Delta (ms)"
    )
    print(
        "-------------------------|-----------------|-------------------|"
        "------------------|------------------"
    )
    for result in (static_result, dynamic_result):
        print(
            f"{result.strategy:24s} | {result.total_ms:15.3f} | "
            f"{result.first_finish_ms:17.3f} | {result.last_finish_ms:16.3f} | "
            f"{result.finish_delta_ms:17.3f}"
        )


if __name__ == "__main__":
    main()
