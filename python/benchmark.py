import time
import statistics
from pathlib import Path

from evaluation import detect_qr


def benchmark_process_directory(image_path: str, iterations: int = 10):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        detect_qr(image_path)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"{'Benchmark':<30}{'Time':>12}{'Iterations':>14}")
    print("-" * 56)
    print(f"{'BM_ProcessDirectory':<30}{mean_time:>9.0f} ms{iterations:>14}")
    print(f"  (stdev: {stdev_time:.1f} ms)")

    return {"mean_ms": mean_time, "stdev_ms": stdev_time, "raw_times": times}


if __name__ == "__main__":
    TEST_DIR = Path(__file__).parent
    image_path = TEST_DIR / "../test_suite/test_integration/image004.jpg"
    benchmark_process_directory(str(image_path.resolve()), iterations=10)
