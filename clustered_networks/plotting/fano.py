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


def _validate_stimulus_curve_shapes(timepoints_s, *curves):
    """Ensure stimulus Fano curves align with provided time axis."""
    timepoints_s = np.asarray(timepoints_s)
    if timepoints_s.ndim != 1:
        raise ValueError(f"timepoints_s must be 1D, got shape {timepoints_s.shape}.")

    for name, curve in curves:
        arr = np.asarray(curve)
        if arr.shape != timepoints_s.shape:
            raise ValueError(
                f"{name} must have shape {timepoints_s.shape}, got {arr.shape}."
            )


def plot_fano_stimulus_response(
    timepoints_s,
    clustered_mean,
    clustered_lower,
    clustered_upper,
    uniform_mean,
    uniform_lower,
    uniform_upper,
    title="Fano Factor Reaction to Higher Input Stimulus",
    figsize=(8, 4.5),
    save_path=None,
):
    """Plot Fano-factor response over time for elevated-input stimulus experiments."""
    _validate_stimulus_curve_shapes(
        timepoints_s,
        ("clustered_mean", clustered_mean),
        ("clustered_lower", clustered_lower),
        ("clustered_upper", clustered_upper),
        ("uniform_mean", uniform_mean),
        ("uniform_lower", uniform_lower),
        ("uniform_upper", uniform_upper),
    )

    timepoints_s = np.asarray(timepoints_s)
    clustered_mean = np.asarray(clustered_mean)
    clustered_lower = np.asarray(clustered_lower)
    clustered_upper = np.asarray(clustered_upper)
    uniform_mean = np.asarray(uniform_mean)
    uniform_lower = np.asarray(uniform_lower)
    uniform_upper = np.asarray(uniform_upper)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        timepoints_s,
        clustered_mean,
        linewidth=1.5,
        label="Clustered, 95% CI",
        color="limegreen",
    )
    ax.fill_between(
        timepoints_s,
        clustered_lower,
        clustered_upper,
        alpha=0.25,
        color="limegreen",
    )
    ax.plot(
        timepoints_s,
        uniform_mean,
        linewidth=1.5,
        label="Uniform, 95% CI",
        color="black",
    )
    ax.fill_between(
        timepoints_s,
        uniform_lower,
        uniform_upper,
        alpha=0.25,
        color="black",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fano Factor")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
