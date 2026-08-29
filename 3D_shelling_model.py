from itertools import product
import matplotlib.pyplot as plt
import numpy as np

EMPTY = 0 

NEIGHBOR_OFFSETS = tuple(
    offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0)
)

def make_random_grid(shape, fraction_empty, rng):
    """Create continuous opinions in (0, 1), with a fraction made empty."""
    grid = rng.random(shape)
    number_empty = int(round(fraction_empty * grid.size))
    empty_indices = rng.choice(grid.size, number_empty, replace=False)
    grid.flat[empty_indices] = EMPTY
    return grid


def shifted(grid, offset):
    """Return one toroidally wrapped neighbour plane."""
    return np.roll(grid, shift=offset, axis=(0, 1, 2))


def consensus_update(grid, consensus_threshold, consensus_weight):
    """Move each opinion toward the mean of compatible occupied neighbours."""
    occupied = grid != EMPTY
    valid_count = np.zeros(grid.shape, dtype=np.int16)
    valid_sum = np.zeros(grid.shape, dtype=float)

    for offset in NEIGHBOR_OFFSETS:
        neighbour = shifted(grid, offset)
        valid = occupied & (neighbour != EMPTY) & (
            np.abs(grid - neighbour) < consensus_threshold
        )
        valid_count += valid
        valid_sum += np.where(valid, neighbour, 0.0)

    updated = grid.copy()
    can_update = occupied & (valid_count > 0)
    neighbour_mean = np.divide(
        valid_sum,
        valid_count,
        out=np.zeros_like(valid_sum),
        where=valid_count > 0,
    )
    updated[can_update] += consensus_weight * (
        neighbour_mean[can_update] - updated[can_update]
    )
    return updated



def similarity_counts(grid, similarity_threshold):
    """Count occupied and similar neighbours for every voxel."""
    occupied = grid != EMPTY
    occupied_count = np.zeros(grid.shape, dtype=np.int16)
    similar_count = np.zeros(grid.shape, dtype=np.int16)

    for offset in NEIGHBOR_OFFSETS:
        neighbour = shifted(grid, offset)
        neighbour_occupied = neighbour != EMPTY
        #counting the number of occupied neighbors per cell
        occupied_count += neighbour_occupied
        similar_count += (
            occupied
            & neighbour_occupied
            & (np.abs(grid - neighbour) < similarity_threshold)
        )

    return similar_count, occupied_count


def segregation_update(grid, similarity_threshold, segregation_threshold, rng):
    """Move unhappy agents to randomly selected empty 3D locations.

    Moves are simultaneous and one-to-one, therefore agents cannot overwrite one another
    and the number of empty locations is always preserved. Unoccupied cells are not counted
    in the agent similarity calculation
    """
    occupied = grid != EMPTY
    similar_count, occupied_count = similarity_counts(grid, similarity_threshold)
    similarity = np.divide(
        similar_count,
        occupied_count,
        out=np.zeros(grid.shape, dtype=float),
        where=occupied_count > 0,
    )
    dissatisfied = np.argwhere(occupied & (similarity < segregation_threshold))
    empty_locations = np.argwhere(~occupied)

    number_moving = min(len(dissatisfied), len(empty_locations))
    if number_moving == 0:
        return grid.copy()

    moving_coords = dissatisfied[rng.permutation(len(dissatisfied))[:number_moving]]
    destinations_coords = empty_locations[rng.permutation(len(empty_locations))[:number_moving]]

    source = tuple(moving_coords.T)
    destination = tuple(destinations_coords.T)

    updated = grid.copy()
    moving_opinions = grid[source].copy()
    updated[source] = EMPTY
    updated[destination] = moving_opinions
    return updated


def plot_grid(ax, grid, title):
    """Plot each occupied cell using coolwarm colour map for opinions."""
    occupied = grid != EMPTY
    coordinates = np.argwhere(occupied)
    opinions = grid[occupied]

    plot = ax.scatter(
        coordinates[:, 2],
        coordinates[:, 1],
        coordinates[:, 0],
        c=opinions,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=100,
        alpha=0.5,
    )
    ax.set(
        title=title,
        xlabel="Width",
        ylabel="Height",
        zlabel="Depth",
    )
    ax.set_box_aspect((grid.shape[2], grid.shape[1], grid.shape[0]))
    return plot


def main():
    grid_shape = (6, 6, 6)
    fraction_empty = 0.2
    number_steps = 100
    similarity_threshold = 0.4
    segregation_threshold = 0.5
    consensus_threshold = 0.8
    consensus_weight = 0.01

    rng = np.random.default_rng(1)
    grid = make_random_grid(grid_shape, fraction_empty, rng)
    initial_grid = grid.copy()

    for _ in range(number_steps):
        grid = consensus_update(grid, consensus_threshold, consensus_weight)
        grid = segregation_update(
            grid, similarity_threshold, segregation_threshold, rng
        )

    occupied = grid != EMPTY
    similar_count, occupied_count = similarity_counts(grid, similarity_threshold)
    similarity = np.divide(
        # we do not count unoccupied cells in the similarity count 
        similar_count,
        occupied_count,
        out=np.zeros(grid.shape, dtype=float),
        where=occupied_count > 0,
    )

    if occupied.any():
        mean_similarity = similarity[occupied].mean()
    else:
        mean_similarity = 1.0

    print(f"Mean similarity after {number_steps} steps: {mean_similarity:.4f}")

    figure = plt.figure(figsize=(12, 6))
    initial_axis = figure.add_subplot(121, projection="3d")
    final_axis = figure.add_subplot(122, projection="3d")
    plot_grid(initial_axis, initial_grid, "Initial grid")
    plot_grid(final_axis, grid, "Grid after simulation")

    plt.show()


if __name__ == "__main__":
    main()

