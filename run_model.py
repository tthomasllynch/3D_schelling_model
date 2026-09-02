from argparse import ArgumentParser, BooleanOptionalAction
import matplotlib.pyplot as plt
import numpy as np

from schelling_3d import (
    make_random_grid,
    consensus_update,
    segregation_update,
    plot_grid,
    mean_similarities,
    plot_graphs
)

GRID_SHAPE = (20, 20, 20)
FRACTION_EMPTY = 0.9
NUM_STEPS = 2000
STEPS_PER_PLOT = 1
DELAY_PER_PLOT = 0.003

SIMILARITY_THRESHOLD = 0.2
SEGREGATION_THRESHOLD = 0.8
CONSENSUS_THRESHOLD = 0.15
CONSENSUS_WEIGHT = 0.06
FLOODFILL_THRESHOLD = 0.2
SEED = 2
EMPTY = 0.0
GRAPHS = False

def parse_args():
    parser = ArgumentParser(description="3D Schelling model of segragation")
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
    if args.graphs:
        rng = np.random.default_rng(args.seed)
        grid = make_random_grid(args.grid_shape, args.fraction_empty, rng)
        initial_grid = grid.copy()
        print(f"Created 3D grid of size {grid.shape} with 26 wrapping neighbours")

        #initialising and positioning figures
        figure = plt.figure(figsize=(20, 10))
        initial_axis = figure.add_subplot(221, projection="3d")
        final_axis = figure.add_subplot(222, projection="3d")

        graph_ax_individual = figure.add_subplot(223)
        graph_ax_global = figure.add_subplot(224)

        plot_grid(initial_axis, initial_grid, "Initial grid")

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

        #Two scatter objects updated every iteration
        individual_points = graph_ax_individual.scatter(
        [],
        [],
        color="#287271",
        s=20,
        )

        global_points = graph_ax_global.scatter(
        [],
        [],
        color="#D1495B",
        s=20,
        )

        steps = []
        individual_values = []
        global_values = []

        graph_ax_individual.set_xlim(0, args.number_steps)
        graph_ax_global.set_xlim(0, args.number_steps)
        
        plt.ion()
        for step in range(args.number_steps):
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

            mean_individual_sim, mean_global_sim = (
                mean_similarities(grid, args.similarity_threshold)
                )

            steps.append(step + 1)
            individual_values.append(mean_individual_sim)
            global_values.append(mean_global_sim)

            print(f"Step {step + 1}/{args.number_steps}:\n"
                f"mean individual similarity = {mean_individual_sim:.4f}\n"
                f"mean global similarity = {mean_global_sim:.4f}\n")
            
            if ((step + 1) % args.steps_per_plot) == 0:

                individual_points.set_offsets(
                np.column_stack((steps, individual_values))
                )
                global_points.set_offsets(
                np.column_stack((steps, global_values))
                )

                final_axis.clear()
                plot_grid(final_axis, grid, f"Grid after step {step + 1}")
                plot_graphs(graph_ax_individual, graph_ax_global, 
                            f"Mean individual similarity after step {step + 1}",
                            f"Mean global similarity after step {step + 1}")
                
                plt.pause(args.delay_per_plot)
        
        print(f"Mean individual similarity after {args.number_steps} steps: {mean_individual_sim:.4f}")
        print(f"Mean global similarity after {args.number_steps} steps: {mean_global_sim:.4f}")

        plt.ioff()
        plt.show()

    else:
        rng = np.random.default_rng(args.seed)
        grid = make_random_grid(args.grid_shape, args.fraction_empty, rng)
        initial_grid = grid.copy()
        print(f"Created 3D grid of size {grid.shape} with 26 wrapping neighbours")

        #initialising and positioning figures
        figure = plt.figure(figsize=(20, 10))
        initial_axis = figure.add_subplot(121, projection="3d")
        final_axis = figure.add_subplot(122, projection="3d")

        plot_grid(initial_axis, initial_grid, "Initial grid")

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
                grid = segregation_update(
                    grid,
                    args.similarity_threshold,
                    args.segregation_threshold,
                    rng,
                )

            mean_individual_sim, mean_global_sim = (
                mean_similarities(grid, args.similarity_threshold)
                )

            print(f"Step {step + 1}/{args.number_steps}:\n"
                f"mean individual similarity = {mean_individual_sim:.4f}\n"
                f"mean global similarity = {mean_global_sim:.4f}\n")
            
            if ((step + 1) % args.steps_per_plot) == 0:

                final_axis.clear()
                plot_grid(final_axis, grid, f"Grid after step {step + 1}")
                plt.pause(args.delay_per_plot)
        
        print(f"Mean individual similarity after {args.number_steps} steps: {mean_individual_sim:.4f}")
        print(f"Mean global similarity after {args.number_steps} steps: {mean_global_sim:.4f}")

        plt.ioff()
        plt.show()




if __name__ == "__main__":
    main()
