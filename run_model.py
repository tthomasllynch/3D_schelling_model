from argparse import ArgumentParser, BooleanOptionalAction
import matplotlib.pyplot as plt
import numpy as np

from schelling_3d import (
    make_random_grid,
    load_grid,
    consensus_update,
    segregation_update,
    mean_similarities,
    find_dissatisfied_agents,
    floodfill,
    opinion_grouping,
)

from plotting_functions import (
    plot_schelling,
    plot_graph,
    plot_floodfill_grid,
    plot_floodfill_graph,
    plot_opinion_group_graph
)

from presets import PRESETS, get_preset

GRID_SHAPE = (20, 20, 20)
FRACTION_EMPTY = 0.9
NUM_STEPS = 200
STEPS_PER_PLOT = 1
EMPTY = 0.0


def parse_args():
    parser = ArgumentParser(description="Schelling's model of segregation applied in 3D")
    parser.add_argument("--presets", choices=PRESETS, default="default")
    parser.add_argument("--consensus", action=BooleanOptionalAction)
    parser.add_argument("--segregation", action=BooleanOptionalAction)
    parser.add_argument("--grid-shape", nargs=3, type=int, default=GRID_SHAPE)
    parser.add_argument("--file", metavar="PATH", help="load a 3D NumPy .npy grid")
    parser.add_argument("--fraction-empty", type=float, default=FRACTION_EMPTY)
    parser.add_argument("--number-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--steps-per-plot", type=int, default=STEPS_PER_PLOT)
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--segregation-threshold", type=float)
    parser.add_argument("--consensus-threshold", type=float)
    parser.add_argument("--consensus-weight", type=float)
    parser.add_argument("--delay-per-plot", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--graphs", action=BooleanOptionalAction, default = False)
    parser.add_argument("--floodfill-threshold", type=float)
    parser.add_argument("--opinion-group-threshold", type=float)
    parser.add_argument("--grouping-graphs", action=BooleanOptionalAction, default = False)
    parser.add_argument( "--show-all-clusters-groups", action="store_true",
        help="show every cluster and opinion group instead of only the 20 largest",
    )

    args = parser.parse_args()
    preset = get_preset(args.presets)

    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if any(size <= 0 for size in args.grid_shape):
        print("Error: grid dimensions must be positive integers")
        exit()

    if not 0 <= args.fraction_empty <= 1:
        print("Error: fraction empty must be between 0 and 1")
        exit()

    if args.number_steps <= 0:
        print("Error: number of steps must be positive")
        exit()

    if args.steps_per_plot <= 0:
        print("Error: steps per plot must be positive")
        exit()

    if args.delay_per_plot < 0:
        print("Error: delay per plot cannot be negative")
        exit()

    if not 0 <= args.similarity_threshold <= 1:
        print("Error: similarity threshold must be between 0 and 1")
        exit()

    if not 0 <= args.segregation_threshold <= 1:
        print("Error: segregation threshold must be between 0 and 1")
        exit()

    if not 0 <= args.consensus_threshold <= 1:
        print("Error: consensus threshold must be between 0 and 1")
        exit()

    if not 0 <= args.consensus_weight <= 1:
        print("Error: consensus weight must be between 0 and 1")
        exit()

    if not 0 <= args.floodfill_threshold <= 1:
        print("Error: floodfill threshold must be between 0 and 1")
        exit()

    if not 0 <= args.opinion_group_threshold <= 1:
        print("Error: opinion group threshold must be between 0 and 1")
        exit()

    return args

def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    if args.file:
        try:
            grid = load_grid(args.file)
        except (OSError, ValueError) as error:
            print(f"Error: could not load grid: {error}")
            exit()
        print(f"Loaded 3D grid of size {grid.shape} from {args.file}")
    else:
        grid = make_random_grid(args.grid_shape, args.fraction_empty, rng)
    initial_grid = grid.copy()

    if args.graphs:
        print(f"Created 3D grid of size {grid.shape} with 26 wrapping neighbours")

        #initialising and positioning figures
        figure = plt.figure(figsize=(20, 8))
        initial_axis = figure.add_subplot(321, projection="3d")
        final_axis = figure.add_subplot(322, projection="3d")

        graph_ax_individual = figure.add_subplot(323)
        graph_ax_global = figure.add_subplot(324)
        graph_ax_dissatisfied = figure.add_subplot(325)
        graph_ax_sd = figure.add_subplot(326)

        steps = []
        individual_mean_values = []
        global_mean_values = []
        dissatisfied_sum_values = []
        sd_values = []

        individual_points = plot_graph(graph_ax_individual, "Mean Individual Similarity", "Similarity", (0, 1))
        global_points = plot_graph(graph_ax_global, "Mean Global Similarity", "Similarity", (0, 1))
        dissatisfied_points = plot_graph(graph_ax_dissatisfied, "Number of Dissatisfied Agents",
                                            "Dissatisfied Agents", (0, np.count_nonzero(grid)))
        sd_points = plot_graph(graph_ax_sd, "Standard Deviation", "Standard Deviation", (0, 1))

        graph_ax_individual.set_xlim(0, args.number_steps)
        graph_ax_global.set_xlim(0, args.number_steps)
        graph_ax_dissatisfied.set_xlim(0, args.number_steps)
        graph_ax_sd.set_xlim(0, args.number_steps)

    else:

        figure = plt.figure(figsize=(20, 10))
        initial_axis = figure.add_subplot(121, projection="3d")
        final_axis = figure.add_subplot(122, projection="3d")

    plot_schelling(initial_axis, initial_grid, "Initial grid")

    figure.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=0, vmax=1),
            cmap="coolwarm",
        ),
        ax=(initial_axis, final_axis),
        label="Opinion",
        shrink=0.7,
        pad=0.1,
    )

    plt.ion()
    for step in range(args.number_steps):
        if args.consensus:
            grid = consensus_update(
                grid,
                args.consensus_threshold,
                args.consensus_weight,
            )

        if args.segregation:
            grid, dissatisfied_count = segregation_update(
                grid,
                args.similarity_threshold,
                args.segregation_threshold,
                rng,
            )
        else:
            # If segregation update is disabled we need to call the
            # find_dissatisfied_agents function
            dissatisfied, _ = find_dissatisfied_agents(
                grid,
                args.similarity_threshold,
                args.segregation_threshold,
            )
            dissatisfied_count = len(dissatisfied)

        mean_individual_sim, mean_global_sim = mean_similarities(
            grid,
            args.similarity_threshold
            )

        if args.graphs:
            occupied = grid != EMPTY
            standard_deviation = np.std(grid[occupied])

            steps.append(step + 1)
            individual_mean_values.append(mean_individual_sim)
            global_mean_values.append(mean_global_sim)
            dissatisfied_sum_values.append(dissatisfied_count)
            sd_values.append(standard_deviation)

        print(f"Step {step + 1}/{args.number_steps}:\n"
        f"mean individual similarity = {mean_individual_sim:.4f}\n"
        f"mean global similarity = {mean_global_sim:.4f}\n"
        f"number of dissatisfied agents = {dissatisfied_count}\n")

        if ((step + 1) % args.steps_per_plot) == 0:

            #scatter objects updated every iteration
            if args.graphs:

                individual_points.set_offsets(
                np.column_stack((steps, individual_mean_values))
                )
                global_points.set_offsets(
                np.column_stack((steps, global_mean_values))
                )
                dissatisfied_points.set_offsets(
                np.column_stack((steps, dissatisfied_sum_values))
                )
                sd_points.set_offsets(
                np.column_stack((steps, sd_values))
                )

            final_axis.clear()
            plot_schelling(final_axis, grid, f"Grid after step {step + 1}")
            plt.pause(args.delay_per_plot)

    print(f"Mean individual similarity after {args.number_steps} steps: {mean_individual_sim:.4f}")
    print(f"Mean global similarity after {args.number_steps} steps: {mean_global_sim:.4f}")
    print(f"Total dissatisfied agents after {args.number_steps} steps: {dissatisfied_count:.4f}")


    if args.grouping_graphs:
        clustered_grid = floodfill(grid, args.floodfill_threshold)
        opinion_groups = opinion_grouping(grid, args.opinion_group_threshold)

        if not args.show_all_clusters_groups:
            cluster_ids, cluster_sizes = np.unique(
                clustered_grid[clustered_grid > 0],
                return_counts=True,
            )
            largest_cluster_ids = cluster_ids[
                np.argsort(cluster_sizes)[::-1][:20]
                ]

            largest_clustered_grid = np.zeros_like(clustered_grid)
            for new_id, original_id in enumerate(largest_cluster_ids, start=1):
                largest_clustered_grid[clustered_grid == original_id] = new_id

            clustered_grid = largest_clustered_grid
            opinion_groups = sorted(
                opinion_groups,
                key=len,
                reverse=True,
                )[:20]

        cluster_figure = plt.figure(figsize=(20, 8), constrained_layout=True)
        cluster_figure.set_constrained_layout_pads(hspace=0.2)
        cluster_figure.suptitle("Numbers above bars show mean opinion")

        final_axis = cluster_figure.add_subplot(221, projection="3d")
        flood_fill_axis = cluster_figure.add_subplot(222, projection="3d")
        cluster_bar_axis = cluster_figure.add_subplot(224)
        opinion_group_axis = cluster_figure.add_subplot(223)

        plot_schelling(final_axis, grid, "Final grid")

        cluster_plot = plot_floodfill_grid(
            flood_fill_axis,
            clustered_grid,
            "Final opinion clusters",
        )
        plot_floodfill_graph(
            cluster_bar_axis,
            grid,
            clustered_grid,
            cluster_plot,
        )
        plot_opinion_group_graph(
            opinion_group_axis,
            opinion_groups,
            "Opinion group sizes"
        )


    plt.ioff()
    plt.show()



if __name__ == "__main__":
    main()
