"""Task 1: single-threaded numerical baseline."""

import os

# These must be set before NumPy is imported. They prevent linked numerical
# libraries from creating worker threads.
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
import subprocess
import time

import numpy as np


STUDENT_ID = 230103024
ARRAY_SIZE = 20_000_000


def cpu_model() -> str:
    """Return the best CPU/platform description available on this OS."""
    model = platform.processor().strip()
    if model:
        return model

    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpu_info:
                for line in cpu_info:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    return platform.machine() or "Unknown"


def main() -> None:
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU model/platform: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")

    rng = np.random.default_rng(STUDENT_ID)
    values = rng.random(ARRAY_SIZE, dtype=np.float64)

    start = time.perf_counter()
    total = np.sum(values, dtype=np.float64)
    standard_deviation = np.std(values, dtype=np.float64)
    elapsed_ms = (time.perf_counter() - start) * 1_000

    print(f"Array elements: {values.size}")
    print(f"Sum: {total:.17g}")
    print(f"Standard deviation: {standard_deviation:.17g}")
    print(f"Wall-clock execution time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
