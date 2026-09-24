import time
import numpy as np

import IPP
import schelling_3d


rng = np.random.default_rng(0)

SHARED_PARAMS = {
    "consensus_threshold": 0.2,
    "consensus_weight": 0.06,
    "similarity_threshold": 0.2,
    "segregation_threshold": 0.75,
    "floodfill_threshold": 0.01,
}


def make_2d_grid():
    """Create a 2D grid with 0.05 fraction empty."""
    grid = rng.random((50, 50))
    empty = rng.choice(grid.size, int(0.05 * grid.size), replace=False)
    grid.flat[empty] = 0
    params = {
        "use_consensus": True,
        "use_segregation": True,
        **SHARED_PARAMS,
    }
    return grid, params


def make_3d_grid():
    """Create a 3D grid with 0.05 fraction empty."""
    grid = rng.random((14, 14, 14))
    empty = rng.choice(grid.size, int(0.05 * grid.size), replace=False)
    grid.flat[empty] = 0
    return grid


def benchmark_2d(num_updates=10):
    total = 0.0
    for _ in range(num_updates):
        grid, params = make_2d_grid()
        start = time.perf_counter()
        grid = IPP.update(grid, params)
        total += time.perf_counter() - start
    return total


def benchmark_3d(num_updates=10):
    total = 0.0
    for _ in range(num_updates):
        grid = make_3d_grid()
        start = time.perf_counter()
        grid, _ = schelling_3d.segregation_update(
            grid,
            SHARED_PARAMS["similarity_threshold"],
            SHARED_PARAMS["segregation_threshold"],
            rng,
        )
        grid = schelling_3d.consensus_update(
            grid,
            SHARED_PARAMS["consensus_threshold"],
            SHARED_PARAMS["consensus_weight"],
        )
        total += time.perf_counter() - start
    return total


def main():
    num_updates = 100

    print("Using shared params:", SHARED_PARAMS)
    two_d_time = benchmark_2d(num_updates)
    three_d_time = benchmark_3d(num_updates)

    print(f"2D IPP total time for {num_updates} updates: {two_d_time:.6f} seconds")
    print(f"3D NumPy total time for {num_updates} updates: {three_d_time:.6f} seconds")

    if three_d_time > 0:
        ratio = two_d_time / three_d_time
        print(f"Speedup factor: {ratio:.2f}x")
    else:
        print("Speedup factor: cannot compute (3D time is zero)")


if __name__ == "__main__":
    main()
