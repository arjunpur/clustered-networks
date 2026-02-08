import numpy as np
import matplotlib.pyplot as plt

from ..analysis.fano import compute_fano_factor


def plot_fano_vs_window(spike_data, window_times_ms=None, save_path=None):
    """Plot Fano factor as a function of analysis window size."""
    if window_times_ms is None:
        window_times_ms = [25, 50, 75, 100, 125, 150, 175, 200]

    uniform_ffs = []
    clustered_ffs = []

    for window_ms in window_times_ms:
        window_t = window_ms / 1000.0
        uniform_ff, clustered_ff = compute_fano_factor(spike_data, window_t=window_t)
        uniform_ffs.append(np.nanmean(uniform_ff))
        clustered_ffs.append(np.nanmean(clustered_ff))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        window_times_ms,
        uniform_ffs,
        "o-",
        color="black",
        linewidth=2,
        markersize=8,
        label="Uniform",
    )
    ax.plot(
        window_times_ms,
        clustered_ffs,
        "s-",
        color="limegreen",
        linewidth=2,
        markersize=8,
        label="Clustered",
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Poisson (FF=1)")
    ax.set_xlabel("Counting Window (ms)")
    ax.set_ylabel("Mean Fano Factor")
    ax.set_title("Fano Factor vs Counting Window Size")
    ax.legend()
    ax.set_xlim(0, max(window_times_ms) + 10)
    ax.set_ylim(0, None)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_fano_vs_ree(R_ee_values, mean_fano_factors, save_path=None):
    """Plot Fano factor as a function of cluster strength R_ee.

    Args:
        R_ee_values: Array of R_ee values tested
        mean_fano_factors: Array of mean Fano factors for each R_ee
        save_path: Path to save figure (optional)

    Returns:
        (fig, ax): Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        R_ee_values,
        mean_fano_factors,
        "o-",
        color="black",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Poisson (FF=1)")

    ax.set_xlabel("$R_{EE}$ (cluster strength)")
    ax.set_ylabel("Mean Fano Factor")
    ax.set_title("Fano Factor vs Cluster Strength")
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
