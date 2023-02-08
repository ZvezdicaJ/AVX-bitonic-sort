from __future__ import annotations

import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from dataclasses import dataclass
from pathlib import Path


ELEMENT_COUNT_REGEX = re.compile(r"(?<=/)[0-9]+")
RESULTS_FILE = (
    Path(os.path.dirname(os.path.realpath(__file__))) / "results/benchmark_results.json"
)


class ElementType(Enum):
    Float = 0
    Double = 1
    Int = 2

    @staticmethod
    def from_str(type: str) -> ElementType:
        return ElementType[type.capitalize()]

    @staticmethod
    def from_benchmark_name(name: str) -> ElementType:
        if "float" in name:
            return ElementType.Float
        elif "double" in name:
            return ElementType.Double
        elif "int" in name:
            return ElementType.Int
        else:
            raise Exception("Unknown element type.")


@dataclass
class Benchmark:
    cpu_time_s: float
    real_time_s: float
    iterations: int
    element_count: int
    element_type: ElementType
    label: str

    def __post_init__(self):
        self.cpu_time_per_iteration_s = self.cpu_time_s / self.iterations


BenchmarkClassification = dict[ElementType, dict[str, list[Benchmark]]]


def parse_benchmarks(
    json_dict: dict,
) -> tuple[BenchmarkClassification, list[Benchmark]]:
    benchmarks: list[Benchmark] = []
    classification: BenchmarkClassification = {}
    for bench in json_dict:

        if bench["run_type"] != "iteration":
            continue

        element_count_match = ELEMENT_COUNT_REGEX.search(bench["name"])

        if element_count_match == None:
            raise Exception(
                f"Failed to parse element count number from {bench['name']}"
            )
        else:
            assert element_count_match is not None
            element_count: int = eval(element_count_match.group(0))

        element_type = ElementType.from_benchmark_name(bench["name"])
        label = bench["name"].split()[0]
        benchmark = Benchmark(
            bench["cpu_time"],
            bench["real_time"],
            bench["iterations"],
            element_count,
            element_type,
            label,
        )
        benchmarks.append(benchmark)
        if element_type not in classification:
            classification[element_type] = {label: [benchmark]}
            continue
        if label not in classification[element_type]:
            classification[element_type][label] = [benchmark]
            continue

        classification[element_type][label].append(benchmark)

    return classification, benchmarks


def load_benchmark_results():
    with RESULTS_FILE.open() as results_file:
        results_json = json.load(results_file)
    return results_json


def plot_benchmarks(classification: BenchmarkClassification):

    fig, axs = plt.subplots(1, len(classification))

    for i, element_type in enumerate(classification):

        std_benchmark_list = classification[element_type]["std"]
        std_cpu_time = np.array(
            [std_benchmark.cpu_time_s for std_benchmark in std_benchmark_list]
        )

        for label in classification[element_type]:
            benchmark_list = classification[element_type][label]
            ax = axs[i]

            element_count = np.array(
                [benchmark.element_count for benchmark in benchmark_list]
            )
            cpu_time = np.array([benchmark.cpu_time_s for benchmark in benchmark_list])

            ax.plot(element_count, cpu_time, label=benchmark_list[0].label)
            ax.set_title(f"{element_type.name} sort timing")
            ax.set_xlabel("element count")
            ax.set_ylabel("cpu time [s]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
    fig.set_tight_layout(True)
    fig.show()
    fig.savefig(f"./results/timing_vs_element_count.jpg")


def plot_speedup(classification: BenchmarkClassification):

    fig, axs = plt.subplots(1, len(classification))

    for i, element_type in enumerate(classification):

        std_benchmark_list = classification[element_type]["std"]
        std_cpu_time = np.array(
            [std_benchmark.cpu_time_s for std_benchmark in std_benchmark_list]
        )
        std_element_count = np.array(
            [benchmark.element_count for benchmark in std_benchmark_list]
        )

        general_benchmark_list = classification[element_type]["general"]
        general_cpu_time = np.array(
            [benchmark.cpu_time_s for benchmark in general_benchmark_list]
        )
        general_element_count = np.array(
            [benchmark.element_count for benchmark in general_benchmark_list]
        )

        ax = axs[i]

        assert (general_element_count == std_element_count).all()
        ax.plot(
            general_element_count,
            std_cpu_time / general_cpu_time,
        )
        ax.set_title(f"{element_type.name} speed-up vs std")
        ax.set_xlabel("element count")
        ax.set_ylabel("speed-up")
        ax.legend()

    fig.set_tight_layout(True)
    fig.show()
    fig.savefig(f"./results/speedup.jpg")


if __name__ == "__main__":
    result_dct = load_benchmark_results()
    benchmarks_dct = result_dct["benchmarks"]
    classification, benchmarks = parse_benchmarks(benchmarks_dct)

    plot_benchmarks(classification)
    plot_speedup(classification)
