from __future__ import annotations

import json
import re
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from dataclasses import dataclass

ELEMENT_COUNT_REGEX = re.compile(r"(?<=/)[0-9]+")


class ElementType(Enum):
    Float = 0
    Double = 1
    Int = 2

    @staticmethod
    def from_str(type: str) -> ElementType:
        return ElementType[str.capitalize()]

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


def parse_benchmarks(json_dict: dict) -> list[Benchmark]:
    benchmarks: list[Benchmark] = []
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

        benchmarks.append(
            Benchmark(
                bench["cpu_time"],
                bench["real_time"],
                bench["iterations"],
                element_count,
                ElementType.from_benchmark_name(bench["name"]),
                bench["name"].split()[0],
            ),
        )
    return benchmarks


def load_benchmark_results():
    with open("./benchmark_results.json", "r") as results_file:
        results_json = json.load(results_file)
    return results_json


def plot_benchmarks(benchmarks: list[Benchmark]):

    std_cpu_time = np.array(
        [benchmark.cpu_time_s for benchmark in benchmarks if benchmark.label == "std"]
    )
    std_element_count = np.array(
        [
            benchmark.element_count
            for benchmark in benchmarks
            if benchmark.label == "std"
        ]
    )

    general_cpu_time = np.array(
        [
            benchmark.cpu_time_s
            for benchmark in benchmarks
            if benchmark.label == "general"
        ]
    )
    general_element_count = np.array(
        [
            benchmark.element_count
            for benchmark in benchmarks
            if benchmark.label == "general"
        ]
    )

    fig, axs = plt.subplots(1, 2)

    axs[0].plot(std_element_count, std_cpu_time, label="std")
    axs[0].plot(general_element_count, general_cpu_time, label="bitonic")

    axs[0].set_title("sort timing vs element count")
    axs[0].set_xlabel("element count")
    axs[0].set_ylabel("cpu time [s]")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].legend()

    assert (std_element_count == general_element_count).all()
    axs[1].plot(std_element_count, std_cpu_time / general_cpu_time)

    axs[1].set_title("sort timing vs element count")
    axs[1].set_xlabel("element count")
    axs[1].set_ylabel("std time / bitonic sort time")
    axs[1].set_xscale("log")
    axs[0].legend()

    fig.show()
    fig.savefig(f"./{benchmarks[0].element_type.name}_benchmarks.jpg")


if __name__ == "__main__":
    result_dct = load_benchmark_results()
    benchmarks_dct = result_dct["benchmarks"]
    benchmarks = parse_benchmarks(benchmarks_dct)

    float_benchmarks = [
        benchmark
        for benchmark in benchmarks
        if benchmark.element_type == ElementType.Float
    ]

    plot_benchmarks(float_benchmarks)
