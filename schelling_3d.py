from itertools import product
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


def load_grid(filename):
    """Load a real-valued 3D NumPy .npy array in z, y, x order."""
    grid = np.load(filename, allow_pickle=False)
    if not isinstance(grid, np.ndarray):
        grid.close()
        raise ValueError("Expected a single NumPy .npy array")
    if grid.ndim != 3 or grid.size == 0:
        raise ValueError("Grid must be a non-empty 3D array")
    if grid.dtype.kind not in "fiu":
        raise ValueError("Grid must contain real numeric opinions")
    if not np.all(np.isfinite(grid) & (grid >= 0) & (grid <= 1)):
        raise ValueError("Opinions must be finite values between 0 and 1")
    return grid.astype(float)


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

    similarity_fraction_grid = np.divide(
    # we do not count unoccupied cells in the similarity count
    # cells with no neighbors have similarity fraction of 0 
    similar_count,
    occupied_count,
    out=np.zeros(grid.shape, dtype=float),
    where=occupied_count > 0,
    )

    return similar_count, occupied_count, similarity_fraction_grid

def mean_similarities(grid, similarity_threshold):
    "Calculate mean individual and global similarities"
    occupied = grid != EMPTY
    similar_count, occupied_count, similarity_fraction_grid = similarity_counts(grid, similarity_threshold)
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
    return mean_individual_similarity, global_similarity

def find_dissatisfied_agents(grid, similarity_threshold, segregation_threshold):
    occupied = grid != EMPTY
    _, _, similarity_fraction_grid = similarity_counts(grid, similarity_threshold)
    
    dissatisfied = np.argwhere(occupied & (similarity_fraction_grid < segregation_threshold))
    empty_locations = np.argwhere(~occupied)

    return dissatisfied, empty_locations


def segregation_update(grid, similarity_threshold, segregation_threshold, rng):
    """Move dissatisfied agents to randomly selected empty 3D locations.

    Agents' moves are determined from the grid before any movement occurs.
    Each agent receives a unique destination, therefore agents cannot overwrite one
    another and the number of empty cells remains unchanged. Empty cells are
    not included when calculating agent similarity.
    """
    dissatisfied, empty_locations = (
        find_dissatisfied_agents(grid, similarity_threshold, segregation_threshold)
        )

    number_moving = min(len(dissatisfied), len(empty_locations))
    if number_moving == 0:
        return grid.copy(), len(dissatisfied)

    moving_coords = dissatisfied[rng.permutation(len(dissatisfied))[:number_moving]]
    destinations_coords = empty_locations[rng.permutation(len(empty_locations))[:number_moving]]

    source = tuple(moving_coords.T)
    destination = tuple(destinations_coords.T)

    updated = grid.copy()
    moving_opinions = grid[source].copy()
    updated[source] = EMPTY
    updated[destination] = moving_opinions
    return updated, len(dissatisfied)


def floodfill(grid, floodfill_threshold):
    """Scan for spatial clusters of occupied cells with similar opinions.

    A cluster grows one layer at a time through all 26 neighbouring positions.
    A neighbouring agent joins when its opinion differs from the current cluster
    mean by less than the floodfill threshold, the mean updates every time a new 
    layer is added. Coordinates wrap at every edge, matching the toroidal 
    neighbourhood used by the rest of the model.

    The returned integer grid contains a cluster number for each clustered cell.
    Empty cells and clusters containing only one agent are labelled as zero. """

    clustered_grid = np.zeros(grid.shape, dtype=np.int32)
    occupied_coords = [
        tuple(coordinate) for coordinate in np.argwhere(grid != EMPTY)
        ]
    visited = set()
    cluster_number = 0

    for starting_coord in occupied_coords:
        if starting_coord in visited:
            continue

        cluster_number += 1
        current_cluster = [starting_coord]
        current_layer = [starting_coord]
        visited.add(starting_coord)

        while len(current_layer) > 0:
            mean_opinion = np.mean([grid[coordinate] for coordinate in current_cluster])
            next_layer = []

            for coordinate in current_layer:
                clustered_grid[coordinate] = cluster_number

                for offset in NEIGHBOR_OFFSETS:
                    neighbour = tuple(
                        (coordinate[axis] + offset[axis]) % grid.shape[axis]
                        for axis in range(grid.ndim)
                    )

                    if neighbour in visited or grid[neighbour] == EMPTY:
                        continue

                    if abs(grid[neighbour] - mean_opinion) < floodfill_threshold:
                        visited.add(neighbour)
                        next_layer.append(neighbour)
                        current_cluster.append(neighbour)

            current_layer = next_layer

    cluster_ids, cluster_sizes = np.unique(
        clustered_grid[clustered_grid > 0], return_counts=True
    )
    single_cluster_ids = cluster_ids[cluster_sizes == 1]
    clustered_grid[np.isin(clustered_grid, single_cluster_ids)] = 0

    return clustered_grid

def opinion_grouping(grid, opinion_group_threshold):
    """Find groups of agents that are similar in opinion
    
    A group starts from the smallest opinion agent, another agent joins when it 
    differs from the first agent by less than the opinion group threshold. A new 
    group is created when every cells that have a difference from the first less
    of less than the opinion group threshold are added to the group.
    
    A list of lists is returned, with each inner list representing a group 
    and containing the opinion of every agent in that group."""

    occupied = grid != EMPTY
    opinions = np.sort(grid[occupied])

    if len(opinions) == 0:
        return []

    groups = [[opinions[0]]]

    for opinion in opinions[1:]:
        current_group = groups[-1]
        minimum_opinion = current_group[0]

        if opinion - minimum_opinion < opinion_group_threshold:
            current_group.append(opinion)
        else:
            groups.append([opinion])

    groups = [group for group in groups if len(group) > 1]

    return groups

