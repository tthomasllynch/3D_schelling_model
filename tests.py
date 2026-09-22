from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from schelling_3d import (
    consensus_update,
    find_dissatisfied_agents,
    floodfill,
    load_grid,
    make_random_grid,
    mean_similarities,
    opinion_grouping,
    segregation_update,
    shifted,
    similarity_counts,
)


def test_make_random_grid():
    rng = np.random.default_rng(1)

    shape = (3, 4, 5)
    grid = make_random_grid(shape, 0, rng)

    assert isinstance(grid, np.ndarray), "Error: grid is not a NumPy array"
    assert grid.shape == shape, "Grid shape does not match the requested shape"
    assert not np.equal(grid, 0).any(), (
        "Grid with fraction_empty of 0 should have no empty cells")

    grid = make_random_grid(shape, 1, rng)
    assert np.equal(grid, 0).all(), (
        "Grid with fraction_empty of 1 should contain only empty cells"
    )

    grid = make_random_grid((10, 10, 10), 0.25, rng)
    empty_count = np.equal(grid, 0).sum()
    assert empty_count == 250, (
    "Empty count should equal 250 with fraction empty 0.25"
    )


def test_shifted():
    grid = [
    [[0, 1], [2, 3]],
    [[4, 5], [6, 7]],
    ]

#                                (z, y, x) 
    shifted_grid = shifted(grid, (1, 0, 0))

    expected = np.array(
        [
            [[4, 5], [6, 7]],
            [[0, 1], [2, 3]],
        ]
    )

    assert np.array_equal(shifted_grid, expected), (
        "A shift of (1, 0, 0) should shift the last depth layer to the front"
    )
    assert np.array_equal(grid, np.arange(8).reshape((2, 2, 2))), (
        "shifted function should not alter the original grid"
    )

def test_consensus_update():
    shape = (3, 3, 3)
    grid = np.full(shape, 0.4)
    updated_grid = consensus_update(grid, 0.1, 0.5)

    assert updated_grid.shape == shape, (
        "Consensus update should not change the grid shape"
    )
    assert np.allclose(grid, updated_grid), (
        "A grid containing one opinion should remain unchanged"
    )

    grid = np.zeros(shape)
    grid[0, 0, 0] = 0.4
    grid[2, 2, 2] = 0.6
    updated_grid = consensus_update(grid, 1.0, 0.5)

    assert np.equal(updated_grid, 0).sum() == 25,(
        "Consensus update should not change the number of empty cells"
    )
    assert updated_grid[0, 0, 0] == 0.5, (
        "The first agent should converge to opinion 0.5"
    )
    assert updated_grid[2, 2, 2] == 0.5, (
        "The second agent should converge to opinion 0.5"
    )


def test_similarity_counts():
    grid = np.full((3, 3, 3), 0.5)
    similar_count, occupied_count, similarity_fraction_grid = similarity_counts(
        grid, 0.1
    )

    assert np.equal(occupied_count, 26).all(), (
        "All agents should have fully occupied neighbors"
    )
    assert np.equal(similar_count, 26).all(), (
        "All agents should have 26 similar neighbors"
    )
    assert np.equal(similarity_fraction_grid, 1).all(), (
        "All agents have a similarity fraction of 1"
    )


def test_mean_similarities():
    grid = np.full((3, 3, 3), 0.5)
    mean_individual, global_similarity = mean_similarities(grid, 0.1)

    assert mean_individual == 1.0, (
        "Mean individual similarity should be 1 for a uniform grid"
    )
    assert global_similarity == 1.0, (
        "Global similarity should be 1 for a uniform grid"
    )

    empty_grid = np.zeros((3, 3, 3))
    mean_individual, global_similarity = mean_similarities(empty_grid, 0.1)

    assert mean_individual == 1.0, (
        "Mean individual similarity should default to 1 for an empty grid"
    )
    assert global_similarity == 1.0, (
        "Global similarity should default to 1 for an empty grid"
    )


def test_find_dissatisfied_agents():
    grid = np.zeros((3, 3, 3))
    grid[0, 0, 0] = 0.1
    grid[2, 2, 2] = 0.9

    dissatisfied, empty_locations = find_dissatisfied_agents(grid, 0.1, 0.5)

    assert len(dissatisfied) == 2, (
        "There should be 2 dissatisfied agents"
    )
    assert len(empty_locations) == 25, (
        "The grid should have 25 empty cells"
    )


def test_segregation_update():
    rng = np.random.default_rng(1)
    grid = np.zeros((3, 3, 3))
    grid[0, 0, 0] = 0.1
    grid[2, 2, 2] = 0.9
    original_grid = grid.copy()

    updated_grid, dissatisfied_count = segregation_update(grid, 0.1, 0.5, rng)

    assert grid.shape == updated_grid.shape, (
        "Segregation update should not change the grid shape"
    )
    assert np.equal(updated_grid, 0).sum() == 25, (
        "Segregation update should not change the number of empty cells"
    )
    assert np.array_equal(
        np.sort(updated_grid[updated_grid != 0]), np.array([0.1, 0.9])
    ), ("Segregation update should preserve every agent opinion")

    assert dissatisfied_count == 2, (
        "The update should report two dissatisfied agents"
    )
    assert np.array_equal(grid, original_grid), (
        "Segregation update should not alter the original grid"
    )
    assert (updated_grid[0, 0, 0] == 0 and updated_grid[2, 2, 2] == 0), (
        "Segregation update should empty both agents' original cells") 


def test_floodfill():
    empty_grid = np.zeros((5, 5, 5))
    clustered_grid = floodfill(empty_grid, 0.05)
    assert np.equal(clustered_grid, 0).all(), (
        "An empty grid should not contain any clusters"
    )

    single_agent_grid = np.zeros((5, 5, 5))
    single_agent_grid[2, 2, 2] = 0.5
    clustered_grid = floodfill(single_agent_grid, 0.05)
    assert np.equal(clustered_grid, 0).all(), (
        "A cluster containing only one agent should be removed"
    )

    grid = np.zeros((5, 5, 5))
    grid[0, 0, 0] = 0.10
    grid[0, 0, 1] = 0.12
    grid[3, 3, 3] = 0.80
    grid[3, 3, 4] = 0.82
    original_grid = grid.copy()

    clustered_grid = floodfill(grid, 0.05)
    first_cluster = clustered_grid[0, 0, 0]
    second_cluster = clustered_grid[3, 3, 3]

    assert first_cluster > 0, "The first connected pair should form a cluster"
    assert clustered_grid[0, 0, 1] == first_cluster, (
        "Connected agents with similar opinions should share a cluster"
    )
    assert second_cluster > 0, "The second connected pair should form a cluster"
    assert clustered_grid[3, 3, 4] == second_cluster, (
        "Connected agents with similar opinions should share a cluster"
    )
    assert first_cluster != second_cluster, (
        "Spatially disconnected groups should form different clusters"
    )
    assert np.count_nonzero(clustered_grid) == 4, (
        "All four agents should belong to one of the two clusters"
    )
    assert np.array_equal(grid, original_grid), (
        "Floodfill should not change the original opinion grid"
    )

    wrapping_grid = np.zeros((5, 5, 5))
    wrapping_grid[0, 2, 2] = 0.4
    wrapping_grid[4, 2, 2] = 0.42
    wrapped_clusters = floodfill(wrapping_grid, 0.05)

    assert wrapped_clusters[0, 2, 2] > 0, (
        "Agents connected across an edge should form a cluster"
    )
    assert wrapped_clusters[0, 2, 2] == wrapped_clusters[4, 2, 2], (
        "Flood fill should use toroidal wrapping at grid boundaries"
    )


def test_opinion_grouping():
    empty_grid = np.zeros((2, 2, 3))
    assert opinion_grouping(empty_grid, 0.05) == [], (
        "An empty grid should not contain any opinion groups"
    )

    grid = np.zeros((2, 2, 3))
    grid[0, 0, 0] = 0.10
    grid[0, 0, 1] = 0.14
    grid[0, 0, 2] = 0.18
    grid[1, 1, 0] = 0.40
    grid[1, 1, 1] = 0.42
    grid[1, 1, 2] = 0.90
    original_grid = grid.copy()

    groups = opinion_grouping(grid, 0.05)

    assert len(groups) == 2, (
        "Only the two opinion groups should be returned"
    )
    assert np.allclose(groups[0], [0.10, 0.14]), (
        "The first two nearby opinions should form the first group"
    )
    assert np.allclose(groups[1], [0.40, 0.42]), (
        "The second pair of nearby opinions should form the second group"
    )
    assert all(len(group) > 1 for group in groups), (
        "Single agent opinion groups should be removed"
    )
    assert np.array_equal(grid, original_grid), (
        "Opinion grouping should not change the original grid"
    )


def test_load_grid():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "grid.npy"
        expected = np.zeros((3, 4, 5))
        expected[0, 1, 2] = 0.25
        expected[1, 2, 3] = 0.9
        np.save(path, expected)
        assert np.array_equal(load_grid(path), expected), (
            "Loading must preserve opinions, axis order, and empty outer layers"
        )

        integer_grid = np.ones((3, 3, 3), dtype=int)
        np.save(path, integer_grid)
        loaded = load_grid(path)
        assert np.array_equal(loaded, integer_grid)
        assert loaded.dtype.kind == "f", "Integer grids must support opinion updates"

        for invalid in (
            np.zeros((3, 3)),
            np.zeros((0, 3, 3)),
            np.full((3, 3, 3), -0.1),
            np.full((3, 3, 3), 1.1),
            np.full((3, 3, 3), np.nan),
            np.full((3, 3, 3), np.inf),
            np.full((3, 3, 3), "0.5"),
            np.ones((3, 3, 3), dtype=complex),
            np.ones((3, 3, 3), dtype=object),
        ):
            np.save(path, invalid)
            try:
                load_grid(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid grid accepted: {invalid.shape}, {invalid.dtype}")

        archive = Path(directory) / "grid.npz"
        np.savez(archive, grid=expected)
        try:
            load_grid(archive)
        except ValueError as error:
            assert "single NumPy .npy array" in str(error)
        else:
            raise AssertionError("Archives must be rejected in favour of a single array")




def run_model_tests():
    test_load_grid()
    print("Load NumPy grid tests passed")
    test_make_random_grid()
    print("Make random grid tests passed")
    test_shifted()
    print("Shifted grid tests passed")
    test_consensus_update()
    print("Consensus update tests passed")
    test_similarity_counts()
    print("Similarity count tests passed")
    test_mean_similarities()
    print("Mean similarity tests passed")
    test_find_dissatisfied_agents()
    print("Dissatisfied agent tests passed")
    test_segregation_update()
    print("Segregation update tests passed")
    test_floodfill()
    print("Flood-fill tests passed")
    test_opinion_grouping()
    print("Opinion grouping tests passed")
    print("All tests passed")


if __name__ == "__main__":
    run_model_tests()
