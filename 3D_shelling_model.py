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