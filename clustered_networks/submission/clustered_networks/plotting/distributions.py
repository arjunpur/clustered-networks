import numpy as np
import matplotlib.pyplot as plt

from ..analysis.firing_rate import compute_firing_rates
from ..analysis.fano import compute_fano_factor


def plot_firing_rate_distribution(
    spike_data, bins=20, rate_range=(0, 15), save_path=None
):
    """Plot firing rate distribution for uniform and clustered networks."""
    uniform_rates, clustered_rates = compute_firing_rates(spike_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        uniform_rates,
        bins=bins,
        histtype="step",
        linewidth=2,
        color="black",
        range=rate_range,
        label=f"Uniform (mean={np.mean(uniform_rates):.2f} Hz)",
    )
    ax.hist(
        clustered_rates,
        bins=bins,
        histtype="step",
        linewidth=2,
        color="limegreen",
        range=rate_range,
        label=f"Clustered (mean={np.mean(clustered_rates):.2f} Hz)",
    )

    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Firing Rate Distribution (Excitatory Neurons)\n"
        f"{spike_data.realizations} realizations x {spike_data.trials} trials"
    )
    ax.legend()

    # Add triangle markers for means
    y_top = ax.get_ylim()[1]
    ax.plot(
        [np.mean(uniform_rates)],
        [y_top * 1.01],
        marker="v",
        markersize=6,
        color="black",
        clip_on=False,
    )
    ax.plot(
        [np.mean(clustered_rates)],
        [y_top * 1.01],
        marker="v",
        markersize=6,
        color="limegreen",
        clip_on=False,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_fano_factor(spike_data, save_path=None):
    """Plot Fano factor distribution for uniform and clustered networks."""
    uniform_ff, clustered_ff = compute_fano_factor(spike_data)

    fig, ax = plt.subplots(figsize=(6, 5))

    bins = np.linspace(0, 3, 35)
    uniform_mean = np.nanmean(uniform_ff)
    clustered_mean = np.nanmean(clustered_ff)

    ax.hist(
        uniform_ff.ravel(),
        bins=bins,
        histtype="step",
        linewidth=2,
        color="black",
        label=f"Uniform (mean={uniform_mean:.3f})",
    )
    ax.hist(
        clustered_ff.ravel(),
        bins=bins,
        histtype="step",
        linewidth=2,
        color="limegreen",
        label=f"Clustered (mean={clustered_mean:.3f})",
    )

    ax.set_xlabel("Fano Factor")
    ax.set_ylabel("Count")
    ax.legend()

    # Add triangle markers for means
    y_top = ax.get_ylim()[1]
    ax.plot(
        [uniform_mean],
        [y_top * 1.01],
        marker="v",
        markersize=6,
        color="black",
        clip_on=False,
    )
    ax.plot(
        [clustered_mean],
        [y_top * 1.01],
        marker="v",
        markersize=6,
        color="limegreen",
        clip_on=False,
    )

    print(f"Mean Fano Factor (Uniform): {uniform_mean:.3f}")
    print(f"Mean Fano Factor (Clustered): {clustered_mean:.3f}")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
