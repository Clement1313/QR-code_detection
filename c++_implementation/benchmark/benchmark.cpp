#include <benchmark/benchmark.h>
#include "detection/pipeline.hh"

static void BM_ProcessDirectory(benchmark::State& state)
{
  for (auto _ : state)
  {
    auto result = qr_code::process_directory(std::string(TEST_DATA_DIR) + "/image004.jpg");
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_ProcessDirectory)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();