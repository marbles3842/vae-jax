import jax.numpy as jnp
import matplotlib.pyplot as plt


def save_samples(samples: jax.Array, n_samples: int, filename: str = "samples.png"):
    ncols = int(n_samples ** 0.5)
    nrows = (n_samples + ncols - 1) // ncols
    padding = 2

    h, w = samples.shape[1], samples.shape[2]
    grid_h = nrows * h + (nrows + 1) * padding
    grid_w = ncols * w + (ncols + 1) * padding
    grid = jnp.zeros((grid_h, grid_w))

    for i in range(n_samples):
        r, c = divmod(i, ncols)
        y = r * (h + padding) + padding
        x = c * (w + padding) + padding
        grid = grid.at[y : y + h, x : x + w].set(samples[i])

    plt.imsave(filename, grid, cmap="gray")