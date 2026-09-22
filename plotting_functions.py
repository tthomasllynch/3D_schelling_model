import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

EMPTY = 0

def plot_floodfill_grid(ax, clustered_grid, title):
    """Plot spatial floodfill clusters in 3D."""
    clustered = clustered_grid > 0
    coordinates = np.argwhere(clustered)
    cluster_numbers = clustered_grid[clustered]

    cluster_plot = ax.scatter(
        coordinates[:, 2],
        coordinates[:, 1],
        coordinates[:, 0],
        c=cluster_numbers,
        cmap="tab20",
        s=100,
        alpha=0.8,
        linewidth=0,
    )

    ax.set(
        title=title,
        xlabel="Width",
        ylabel="Height",
        zlabel="Depth",
        xlim=(0, clustered_grid.shape[2] - 1),
        ylim=(0, clustered_grid.shape[1] - 1),
        zlim=(0, clustered_grid.shape[0] - 1),
    )
    ax.set_box_aspect(
        (
            clustered_grid.shape[2],
            clustered_grid.shape[1],
            clustered_grid.shape[0],
        )
    )

    return cluster_plot


def plot_floodfill_graph(ax, grid, clustered_grid, cluster_plot):
    """Plot spatial cluster sizes with mean opinions above the bars."""
    cluster_ids = np.unique(clustered_grid[clustered_grid > 0])
    cluster_sizes = [
        np.count_nonzero(clustered_grid == cluster_id)
        for cluster_id in cluster_ids
    ]
    cluster_mean_opinions = [
        grid[clustered_grid == cluster_id].mean()
        for cluster_id in cluster_ids
    ]
    cluster_colours = cluster_plot.cmap(cluster_plot.norm(cluster_ids))

    bars = ax.bar(
        cluster_ids,
        cluster_sizes,
        color=cluster_colours,
    )
    ax.bar_label(
        bars,
        labels=[f"{opinion:.2f}" for opinion in cluster_mean_opinions],
        padding=3,
    )
    ax.set(
        title="Spatial cluster sizes",
        xlabel="Cluster number",
        ylabel="Number of agents",
        ylim=(0, max(1, max(cluster_sizes, default=0) * 1.15)),
        xticks=cluster_ids,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.5,
    )

    return bars

def plot_opinion_group_graph(ax, groups, title):
    """Plot opinion group sizes with mean opinions above the bars."""

    group_sizes = [len(group) for group in groups]
    group_mean_opinions = [np.mean(group) for group in groups]

    group_numbers = np.arange(1, len(groups) + 1)
    group_colours = plt.cm.coolwarm(group_mean_opinions)

    bars = ax.bar(
        group_numbers,
        group_sizes,
        color=group_colours,
    )
    ax.bar_label(
        bars,
        labels=[f"{opinion:.2f}" for opinion in group_mean_opinions],
        padding=3,
    )

    ax.set(
        title=title,
        xlabel="Opinion group number",
        ylabel="Number of agents",
        ylim=(0, max(1, max(group_sizes, default=0) * 1.15)),
        xticks=group_numbers,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.5,
    )

    return bars

def plot_schelling(ax, grid, title):
    """Plot each occupied cell using coolwarm colour map for opinions."""
    occupied = grid != EMPTY
    coordinates = np.argwhere(occupied)
    opinions = grid[occupied]

    ax.scatter(
        coordinates[:, 2],
        coordinates[:, 1],
        coordinates[:, 0],
        c=opinions,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        marker="o",
        s=90,
        alpha=0.9,
        depthshade=True,
        edgecolors=(0, 0, 0, 0.25),
        linewidths=0.25,
    )
    ax.set(
        title=title,
        xlabel="Width",
        ylabel="Height",
        zlabel="Depth",
        xlim=(0, grid.shape[2] - 1),
        ylim=(0, grid.shape[1] - 1),
        zlim=(0, grid.shape[0] - 1),
    )
    ax.set_box_aspect((grid.shape[2], grid.shape[1], grid.shape[0]))


def plot_graph(graph_ax, title, y_label, y_lim):

    graph_points = graph_ax.scatter(
    [],
    [],
    color="#287271",
    s=5,)

    graph_ax.set(
    title= title,
    xlabel="Step",
    ylabel= y_label,
    ylim=y_lim,
    )

    graph_ax.grid(
    True, 
    linestyle="--", 
    linewidth=0.6, 
    alpha=0.5,
    )

    return graph_points
