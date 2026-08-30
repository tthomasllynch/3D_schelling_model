from argparse import ArgumentParser, BooleanOptionalAction
import matplotlib.pyplot as plt
import numpy as np

from schelling_3d import (
    make_random_grid,
    consensus_update,
    similarity_counts,
    segregation_update,
    plot_grid,
)

GRID_SHAPE = (10,10,10)
FRACTION_EMPTY = 0.9
NUM_STEPS = 100
STEPS_PER_PLOT = 1
DELAY_PER_PLOT = 0.3

SIMILARITY_THRESHOLD = 0.1
SEGREGATION_THRESHOLD = 0.7
CONSENSUS_THRESHOLD = np.clip(2 * SIMILARITY_THRESHOLD, 0, 1)
CONSENSUS_WEIGHT = 0.06
FLOODFILL_THRESHOLD = 0.2
SEED = 1
EMPTY = 0.0

def parse_args():
    parser = ArgumentParser(description="3D version of the CPA opinion model")
    parser.add_argument("--consensus", action=BooleanOptionalAction, default=True)
    parser.add_argument("--segregation", action=BooleanOptionalAction, default=True)
    parser.add_argument("--grid-shape", nargs=3, type=int, default=GRID_SHAPE)
    parser.add_argument("--fraction-empty", type=float, default=FRACTION_EMPTY)
    parser.add_argument("--number-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--segregation-threshold", type=float, default=SEGREGATION_THRESHOLD)
    parser.add_argument("--consensus-threshold", type=float, default=CONSENSUS_THRESHOLD)
    parser.add_argument("--consensus-weight", type=float, default=CONSENSUS_WEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()

def main():
    args = parse_args()
    #(x, y , z)
    rng = np.random.default_rng(args.seed)
    grid = make_random_grid(args.grid_shape, args.fraction_empty, rng)
    initial_grid = grid.copy()

    for _ in range(args.number_steps):
        if args.consensus:
            grid = consensus_update(
                grid,
                args.consensus_threshold,
                args.consensus_weight,
            )

        if args.segregation:
            grid = segregation_update(
                grid,
                args.similarity_threshold,
                args.segregation_threshold,
                rng,
            )

    occupied = grid != EMPTY
    similar_count, occupied_count, similarity_fraction_grid = similarity_counts(grid, args.similarity_threshold)
    total_occupied_neighbours = occupied_count[occupied].sum()

    if occupied.any():
        mean_individual_similarity = similarity_fraction_grid[occupied].mean()
    else:
        mean_individual_similarity = 1.0

    if total_occupied_neighbours > 0:
        global_similarity = (
            similar_count[occupied].sum()
            /total_occupied_neighbours
        )
    else:
        global_similarity = 1.0
    
    print(f"Mean individual similarity after {args.number_steps} steps: {mean_individual_similarity:.4f}")
    print(f"Mean global similarity after {args.number_steps} steps: {global_similarity:.4f}")
    figure = plt.figure(figsize=(12, 6))
    initial_axis = figure.add_subplot(121, projection="3d")
    final_axis = figure.add_subplot(122, projection="3d")
    plot_grid(initial_axis, initial_grid, "Initial grid")
    colour_plot = plot_grid(final_axis, grid, "Grid after simulation")
    figure.colorbar(
        colour_plot,
        ax=(initial_axis, final_axis),
        label="Opinion",
        shrink=0.7,
        pad=0.1,
    )
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
