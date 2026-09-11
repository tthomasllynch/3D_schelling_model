from argparse import ArgumentParser, BooleanOptionalAction
import matplotlib.pyplot as plt
import numpy as np

from schelling_3d import (
    make_random_grid,
    consensus_update,
    segregation_update,
    plot_schelling,
    mean_similarities,
    find_dissatisfied_agents,
    plot_graph
)

GRID_SHAPE = (20, 20, 20)
FRACTION_EMPTY = 0.9
NUM_STEPS = 400
STEPS_PER_PLOT = 1
DELAY_PER_PLOT = 0.000000001

SIMILARITY_THRESHOLD = 0.2
SEGREGATION_THRESHOLD = 20/26
CONSENSUS_THRESHOLD = 0.17
CONSENSUS_WEIGHT = 0.06
FLOODFILL_THRESHOLD = 0.2
SEED = 2
EMPTY = 0.0
GRAPHS = False

def parse_args():
    parser = ArgumentParser(description="3D Schelling model of segregation")
    parser.add_argument("--consensus", action=BooleanOptionalAction, default=True)
    parser.add_argument("--segregation", action=BooleanOptionalAction, default=True)
    parser.add_argument("--grid-shape", nargs=3, type=int, default=GRID_SHAPE)
    parser.add_argument("--fraction-empty", type=float, default=FRACTION_EMPTY)
    parser.add_argument("--number-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--steps-per-plot",type=int,default=STEPS_PER_PLOT)
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--segregation-threshold", type=float, default=SEGREGATION_THRESHOLD)
    parser.add_argument("--consensus-threshold", type=float, default=CONSENSUS_THRESHOLD)
    parser.add_argument("--consensus-weight", type=float, default=CONSENSUS_WEIGHT)
    parser.add_argument("--delay-per-plot", type=float, default=DELAY_PER_PLOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--graphs", action=BooleanOptionalAction, default=GRAPHS)
    return parser.parse_args()

def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    grid = make_random_grid(args.grid_shape, args.fraction_empty, rng)
    initial_grid = grid.copy()

    if args.graphs:
        print(f"Created 3D grid of size {grid.shape} with 26 wrapping neighbours")

        #initialising and positioning figures
        figure = plt.figure(figsize=(20, 10))
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

    plt.ioff()
    plt.show()
    


if __name__ == "__main__":
    main()
