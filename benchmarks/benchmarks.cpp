#include <BitonicSort.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <BitonicSortCommon.h>
#include <RandomVectorGenerator.h>

namespace bitonic_sort::benchmarks {

using utils::getRandomVector;

namespace {
constexpr std::size_t maxVectorSize = 1 << 27; // std::numeric_limits<std::int32_t>::max();

// Helper macro to obtain correct function pointer.
// clang-format off
#define LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(FuncName, ElementType)     \
    [](std::span<ElementType> span) {                                    \
        FuncName(span);                                                   \
    }
// clang-format on

// for the following two benches select values manually
// clang-format off
static void CustomArguments(benchmark::internal::Benchmark *b) {
    std::vector<std::int64_t> benchesToRun(
        {10,       20,      40,      50,      80,      100,      200,
         400,      800,     1600,    3200,    5000,    7500,     10000,
         12000,    15000,   17000,   20000,   25000,   27000,    30000,
         35000,    40000,   50000,   70000,   100000,  120000,   150000,
         200000,   400000,  600000,  800000,  1000000, 1200000,  1500000,
         1800000,  2000000, 2500000, 3000000, 5000000, std::size_t(7e6), 
         std::size_t(1e7), std::size_t(2e7), std::size_t(5e7)});
    // clang-format on
    for (auto const &i : benchesToRun) {
        b->Args({i});
    }
}

template <typename T>
void bitonicSortBenchmark(benchmark::State &state, std::function<void(std::span<T>)> functor) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<T> vec = getRandomVector<T>(state.range(0));
        state.ResumeTiming();
        // benchmark::DoNotOptimize(sort_2n_vector(vec.data(), 0,
        // vec.size() - 1)); // DoNoOptimize will store the result
        // to the memory
        functor(vec);
    }
    state.counters["vector size:"] = state.range(0);
    state.SetComplexityN(state.range(0));
}

constexpr static auto bitonicSortBenchmarkF = bitonicSortBenchmark<float>;
constexpr static auto bitonicSortBenchmarkD = bitonicSortBenchmark<double>;

#define DEFINE_BENCHMARK(BenchmarkTemplate, functionToTest, SortElementType, Name)            \
    BENCHMARK_CAPTURE(BenchmarkTemplate,                                                      \
                      Name,                                                                   \
                      LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(functionToTest, SortElementType))

// float benchmarks
DEFINE_BENCHMARK(bitonicSortBenchmarkF, sort_2n, float, sort_2n_float_benchmark)
    ->RangeMultiplier(2)
    ->Range(4, maxVectorSize)
    ->Complexity(benchmark::oN)
    ->Name("2n float");

DEFINE_BENCHMARK(bitonicSortBenchmarkF, sort_8n, float, sort_8n_float_benchmark)
    ->RangeMultiplier(4)
    ->Range(8, maxVectorSize)
    ->Complexity(benchmark::oN)
    ->Name("8n float");

DEFINE_BENCHMARK(bitonicSortBenchmarkF, sort, float, sort_float_benchmark)
    ->Apply(CustomArguments)
    ->Complexity(benchmark::oN)
    ->Name("general float");

DEFINE_BENCHMARK(bitonicSortBenchmarkF, std::ranges::sort, float, sort_float_benchmark)
    ->Apply(CustomArguments)
    ->Complexity(benchmark::oN)
    ->Name("std float");

// double benchmarks
DEFINE_BENCHMARK(bitonicSortBenchmarkD, sort_2n, double, sort_2n_float_benchmark)
    ->RangeMultiplier(2)
    ->Range(4, maxVectorSize)
    ->Complexity(benchmark::oN)
    ->Name("2n double");

DEFINE_BENCHMARK(bitonicSortBenchmarkD, sort_4n, double, sort_8n_float_benchmark)
    ->RangeMultiplier(4)
    ->Range(8, maxVectorSize)
    ->Complexity(benchmark::oN)
    ->Name("4n double");

DEFINE_BENCHMARK(bitonicSortBenchmarkD, sort, double, sort_float_benchmark)
    ->Apply(CustomArguments)
    ->Complexity(benchmark::oN)
    ->Name("general double");

DEFINE_BENCHMARK(bitonicSortBenchmarkD, std::ranges::sort, double, sort_float_benchmark)
    ->Apply(CustomArguments)
    ->Complexity(benchmark::oN)
    ->Name("std double");

/*BENCHMARK_CAPTURE(bitonicSortBenchmark<float>,
                  LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(sort_8n, float))
    ->RangeMultiplier(2)
    ->Range(8, maxVectorSize)
    ->Complexity(benchmark::oN);

BENCHMARK_CAPTURE(bitonicSortBenchmark<double>,
                  LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(sort_2n, double))
    ->RangeMultiplier(2)
    ->Range(4, maxVectorSize)
    ->Complexity(benchmark::oN);

BENCHMARK_CAPTURE(bitonicSortBenchmark<double>,
                  LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(sort_4n, double))
    ->RangeMultiplier(4)
    ->Range(4, maxVectorSize)
    ->Complexity(benchmark::oN);

BENCHMARK_CAPTURE(bitonicSortBenchmark<float>,
                  LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(sort, float))
    ->Apply(CustomArguments);

BENCHMARK_CAPTURE(bitonicSortBenchmark<double>,
                  LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(sort_2n, double))
    ->Apply(CustomArguments);

BENCHMARK_CAPTURE(bitonicSortBenchmark<float>,
                  LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(std::ranges::sort,
                                                        float))
    ->Apply(CustomArguments);

BENCHMARK_CAPTURE(bitonicSortBenchmark<double>,
                  LAMBDA_WRAPPER_FOR_BENCHMARK_FUNCTION(std::ranges::sort,
                                                        double))
    ->Apply(CustomArguments);*/

} // namespace
} // namespace bitonic_sort::benchmarks

/*static void std_double_sort_bench(benchmark::State &state) {

    for (auto _ : state) {
        state.PauseTiming();
        aligned_vector<double> vec;
        vec.reserve(state.range(0));
        for (int i = 0; i < state.range(0); i++)
            vec.push_back(random_double());
        state.ResumeTiming();
        // benchmark::DoNotOptimize(sort_2n_vector(vec.data(), 0,
        // vec.size() - 1)); // DoNoOptimize will store the result
        // to the memory
        std::sort(std::begin(vec), std::end(vec));
    }
    state.counters["Number to sort:"] = state.range(0);
    state.SetComplexityN(state.range(0));
}*/

/*
BENCHMARK(std_float_sort_bench)
    ->RangeMultiplier(2)
    ->Range(2, 1e8)
    ->RangeMultiplier(3)
    ->Range(3, 1e8)
    ->Complexity(benchmark::oN);
*/

// BENCHMARK(std_double_sort_bench)->Apply(CustomArguments);

// KEY VALUE SORT
/*
static void bitonic_2n_double_key_value_sort_bench(benchmark::State &state)
{

    for (auto _ : state) {
        state.PauseTiming();
        aligned_vector<double> vec;
        vec.reserve(state.range(0));
        aligned_vector<int> indices;
        indices.reserve(state.range(0));
        for (int i = 0; i < state.range(0); i++) {
            vec.push_back(random_double());
            indices.push_back(i);
        }
        state.ResumeTiming();
        // benchmark::DoNotOptimize(sort_2n_vector(vec.data(), 0,
        // vec.size() - 1)); // DoNoOptimize will store the result
        // to the memory
        BITONIC_SORT_KEY_VALUE::sort_2n_key_value(vec.data(),
indices.data(), vec.size());
    }
    state.counters["Number to sort:"] = state.range(0);
    state.SetComplexityN(state.range(0));
}
BENCHMARK(bitonic_2n_double_key_value_sort_bench)
    ->RangeMultiplier(2)
    ->Range(8, 67108864)
    ->Complexity(benchmark::oN);
;

static void bitonic_4n_double_key_value_sort_bench(benchmark::State &state)
{

    for (auto _ : state) {
        state.PauseTiming();

        aligned_vector<double> vec;
        vec.reserve(state.range(0));

        aligned_vector<int> indices;
        indices.reserve(state.range(0));

        for (int i = 0; i < state.range(0); i++) {
            vec.push_back(random_double());
            indices.push_back(i);
        }
        state.ResumeTiming();
        // benchmark::DoNotOptimize(sort_2n_vector(vec.data(), 0,
        // vec.size() - 1)); // DoNoOptimize will store the result
        // to the memory
        BITONIC_SORT_KEY_VALUE::sort_4n_key_value(vec.data(),
indices.data(), vec.size());
    }
    state.counters["Number to sort:"] = state.range(0);
    state.SetComplexityN(state.range(0));
}

BENCHMARK(bitonic_4n_double_key_value_sort_bench)
    ->RangeMultiplier(4)
    ->Range(8, 67108864)
    ->Complexity(benchmark::oN);
;

static void bitonic_double_key_value_sort_bench(benchmark::State &state) {

    for (auto _ : state) {
        state.PauseTiming();
        aligned_vector<double> vec;
        vec.reserve(state.range(0));
        aligned_vector<int> indices;
        indices.reserve(state.range(0));
        for (int i = 0; i < state.range(0); i++) {
            vec.push_back(random_double());
            indices.push_back(i);
        }
        state.ResumeTiming();
        // benchmark::DoNotOptimize(sort_2n_vector(vec.data(), 0,
        // vec.size() - 1)); // DoNoOptimize will store the result
        // to the memory
        BITONIC_SORT_KEY_VALUE::sort_key_value(vec.data(), indices.data(),
                                               vec.size());
    }
    state.counters["Number to sort:"] = state.range(0);
    state.SetComplexityN(state.range(0));
}

BENCHMARK(bitonic_double_key_value_sort_bench)->Apply(CustomArguments);
*/