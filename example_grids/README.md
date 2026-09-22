# Example 3D grids

Each `.npy` file stores a 3D NumPy array indexed as `grid[z, y, x]`.
Opinion `0` is an empty cell; occupied-cell opinions are greater than 0 and at most 1.

- `hollow_cube.npy`: an alternating `0.25`/`0.75` cube shell.
- `starburst.npy`: 26 rays with opinions ranging from `0.1` to `0.9`.
- `double_helix.npy`: two winding strands with low and high opinions.
- `layered_sphere.npy`: a filled sphere with a smooth opinion gradient.

Run one from the repository directory, for example:

```bash
python3 run_model.py --file example_grids/starburst.npy
```

Each file preserves its full grid dimensions, including empty outer layers.
Inspect an array in Python with `grid = np.load("example_grids/starburst.npy", allow_pickle=False)`.
