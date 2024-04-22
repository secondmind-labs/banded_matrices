"""A script to perform serious benchmarking of broadcasted ops

Usage: First run pytest and output benchmarked data. Then run this script as main,
to generate results.

1. pytest -k broadcasting_profile --benchmark-autosave --benchmark-json ./output.json -v
2. python test_run_full_broadcasting_profile.py ./output.json
"""
# pylint: disable=redefined-outer-name,cell-var-from-loop
import argparse
import json
import math
from collections import defaultdict
from typing import Callable, List, NamedTuple, Tuple

import numpy as np
import pytest

from tests.utils.banded_matrices_utils import generate_banded_tensor

from ..perf.test_unary_broadcast import (
    UNARY_BAND_OPS,
    broadcast_unary_using_native,
    broadcast_unary_using_py_broadcast,
)

LEADING_DIMS_EXP = {
    "single_scale": [[10], [50], [100], [200], [500], [700], [1000]],
    "nesting": [[500], [5, 100], [100, 5], [5, 10, 10], [5, 5, 5, 2, 2]],
}
DIMENSION_EXP = [5, 10, 50, 100, 200, 500, 700, 1000]
LOWER_EXP = [3, 5, 7]


class BenchmarkedItem(NamedTuple):
    """Each instance represents a timed run on benchmark.pedantic()"""

    experiment: str
    func: Callable
    chol_dim: int = 200  # Dim (No of points)
    lower: int = 3  # Lower BW (matern 3/2 = 3)
    upper: int = 0  # Assume none
    leading_dims: Tuple[int] = (100,)  # Batch size


class Experiment(NamedTuple):
    """A series of experiments varying an x variable"""

    name: str
    x_variable: str
    benchmarks: List[BenchmarkedItem]


@pytest.fixture
def data_loader():
    """We want to benchamark always on the same matrix, to save time"""
    DATA_CACHE = {}

    def load(leading_dims, chol_dim, lower, upper):
        data_key = (leading_dims, chol_dim, lower, upper)
        if data_key in DATA_CACHE:
            return DATA_CACHE[data_key]
        else:
            flat_unary = generate_banded_tensor(
                (np.product(leading_dims), lower, upper, chol_dim),
                ensure_positive_definite=True,
            )
            unary = flat_unary.reshape(leading_dims + flat_unary.shape[1:])
            DATA_CACHE[data_key] = unary
            return unary

    return load


def get_experiments():
    """Generate a list of experiments."""
    experiments = []
    experiments.append(
        Experiment(
            name="single_batch",
            x_variable="leading_dims",
            benchmarks=[
                BenchmarkedItem(f"single_batch-{b}", f, leading_dims=tuple(b))
                for b in LEADING_DIMS_EXP["single_scale"]
                for f in UNARY_BAND_OPS
            ],
        )
    )
    experiments.append(
        Experiment(
            name="dim_size",
            x_variable="chol_dim",
            benchmarks=[
                BenchmarkedItem("dim_size", f, chol_dim=n)
                for n in DIMENSION_EXP
                for f in UNARY_BAND_OPS
            ],
        )
    )
    experiments.append(
        Experiment(
            name="multi_batch",
            x_variable="leading_dims",
            benchmarks=[
                BenchmarkedItem(f"multi_batch-{b}-", f, leading_dims=tuple(b))
                for b in LEADING_DIMS_EXP["nesting"]
                for f in UNARY_BAND_OPS
            ],
        )
    )
    experiments.append(
        Experiment(
            name="single_batch",
            x_variable="leading_dims",
            benchmarks=[
                BenchmarkedItem("lower-dim", f, lower=l)
                for l in LOWER_EXP
                for f in UNARY_BAND_OPS
            ],
        )
    )
    return experiments


def get_benchmarked_exps():
    """Convert experiments to a list of benchmarked runs"""
    return [b for e in get_experiments() for b in e.benchmarks]


@pytest.mark.skip(reason="Very Slow!! Runs all profiling experiments")
@pytest.mark.parametrize(
    "name, func, chol_dim, lower, upper, leading_dims", get_benchmarked_exps()
)
def test_map_fn(benchmark, data_loader, name, func, chol_dim, lower, upper, leading_dims):
    """
    A comparison between several ways to apply the same operation on a stack of banded matrices.
    """
    data = data_loader(leading_dims, chol_dim, lower, upper)
    benchmark.pedantic(
        broadcast_unary_using_py_broadcast, args=[func, True, data], rounds=10, iterations=10
    )


@pytest.mark.skip(reason="Very Slow!! Runs all profiling experiments")
@pytest.mark.parametrize(
    "name, func, chol_dim, lower, upper, leading_dims", get_benchmarked_exps()
)
def test_native_perf(benchmark, data_loader, name, func, chol_dim, lower, upper, leading_dims):
    """
    A comparison between several ways to apply the same operation on a stack of banded matrices.
    """
    data = data_loader(leading_dims, chol_dim, lower, upper)

    benchmark.pedantic(
        broadcast_unary_using_native, args=[func, True, data], rounds=10, iterations=10
    )


def load_results(path):
    """Load the results from the given json at the path"""
    with open(path, "r") as f:
        report_data = json.load(f)
    benchmarks = report_data["benchmarks"]

    # str -> [ops -> [eval_method -> [stats]]]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for b in benchmarks:
        params = b["params"]
        exp = params["name"].split("-")[0]
        op = params["func"].split(" ")[1]
        eval_method = b["name"].split("[")[0]
        stats = b["stats"]
        results[exp][op][eval_method].append((params, stats))
    return results


def plot_results(results):
    """Plot one figure per experiment, with each op in its own subplot"""
    experiments = {e.name: e for e in get_experiments()}

    for exp, ops in results.items():
        rows = 2
        cols = math.ceil(len(ops.keys()) / rows)
        fig, axes = plt.subplots(figsize=(16, 8), nrows=rows, ncols=cols)
        fig.suptitle(f"x_axis: {exp}")

        def get_x_y_value(i, params, stats):
            x = i
            if exp in experiments:
                x = params[experiments[exp].x_variable]
            y = stats["median"]
            ly = stats["ld15iqr"]
            uy = stats["hd15iqr"]
            return x, y, ly, uy

        for i, (op, op_lines) in enumerate(ops.items()):
            r = i // cols
            c = i % cols

            rs = []
            for l, line_list in op_lines.items():
                xs, ys, ly, uy = zip(
                    *sorted(
                        [get_x_y_value(i, p, s) for i, (p, s) in enumerate(line_list)],
                        key=lambda x: [0],
                    )
                )
                rs.append(np.array(ys))
                axes[r][c].errorbar(
                    xs,
                    ys,
                    yerr=(ly, uy),
                    elinewidth=0.7,
                    capsize=0.7,
                    label=l.replace("test_", ""),
                )
            mean_speedup = np.mean(rs[0][-4:] / rs[1][-4:])
            axes[r][c].set_title(f"{op}: {mean_speedup:.2f}x")
            axes[r][c].set_ylabel("secs")
            axes[r][c].legend()
            axes[r][c].grid(True, which="both")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot graphs of profiling experiments.")
    from matplotlib import pyplot as plt

    plt.rc("grid", linestyle="dotted")
    parser.add_argument("--json", type=str, help="path to json file")
    args = vars(parser.parse_args())
    test_results = load_results(args["json"])
    plot_results(test_results)
