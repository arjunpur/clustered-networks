import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ..analysis.correlation import compute_correlation_coefficients
from ..analysis.tail import analyze_correlation_tail


def plot_correlation_all_pairs(
    spike_data, corr_windows=None, corr_step=0.025, save_path=None
):
    """Plot correlation coefficient distribution for all neuron pairs.

    Args:
        spike_data: SpikeData object
        corr_windows: List of window sizes in seconds (default: [0.05, 0.1])
        corr_step: Step size in seconds (default: 0.025 = 25ms)
        save_path: Path to save figure
    """
    if corr_windows is None:
        corr_windows = [0.05, 0.1]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Colors for each window: (uniform_color, clustered_color)
    window_colors = [
        ("black", "limegreen"),
        ("gray", "dodgerblue"),
        ("darkred", "orange"),
        ("purple", "cyan"),
    ]

    bins = np.linspace(-0.5, 0.5, 50)

    for w_idx, corr_window in enumerate(corr_windows):
        window_ms = int(corr_window * 1000)
        uniform_color, clustered_color = window_colors[w_idx % len(window_colors)]

        uniform_coeffs, clustered_coeffs = compute_correlation_coefficients(
            spike_data, corr_window, corr_step, same_cluster_only=False
        )

        ax.hist(
            uniform_coeffs,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=uniform_color,
            label=f"Uniform ({window_ms}ms)",
        )
        ax.hist(
            clustered_coeffs,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=clustered_color,
            label=f"Clustered ({window_ms}ms)",
        )

        print(f"Window {window_ms}ms (all pairs):")
        print(
            f"  Uniform: {np.nanmean(uniform_coeffs):.4f}, "
            f"Clustered: {np.nanmean(clustered_coeffs):.4f}"
        )

    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Count")
    ax.set_xlim(-0.5, 0.5)
    ax.legend()

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((6, 6))
    ax.yaxis.set_major_formatter(fmt)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_correlation_same_cluster(
    spike_data, corr_windows=None, corr_step=0.025, save_path=None
):
    """Plot correlation coefficient distribution for same-cluster neuron pairs.

    Args:
        spike_data: SpikeData object
        corr_windows: List of window sizes in seconds (default: [0.05, 0.1])
        corr_step: Step size in seconds (default: 0.025 = 25ms)
        save_path: Path to save figure
    """
    if corr_windows is None:
        corr_windows = [0.05, 0.1]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Colors for each window: (uniform_color, clustered_color)
    window_colors = [
        ("black", "limegreen"),
        ("gray", "dodgerblue"),
        ("darkred", "orange"),
        ("purple", "cyan"),
    ]

    bins = np.linspace(-0.5, 1.0, 50)

    for w_idx, corr_window in enumerate(corr_windows):
        window_ms = int(corr_window * 1000)
        uniform_color, clustered_color = window_colors[w_idx % len(window_colors)]

        uniform_coeffs, clustered_coeffs = compute_correlation_coefficients(
            spike_data, corr_window, corr_step, same_cluster_only=True
        )

        ax.hist(
            uniform_coeffs,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=uniform_color,
            label=f"Uniform ({window_ms}ms)",
        )
        ax.hist(
            clustered_coeffs,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=clustered_color,
            label=f"Clustered ({window_ms}ms)",
        )

        print(f"Window {window_ms}ms (same cluster):")
        print(
            f"  Uniform: {np.nanmean(uniform_coeffs):.4f}, "
            f"Clustered: {np.nanmean(clustered_coeffs):.4f}"
        )

    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Count")
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(0, 15e4)
    ax.legend()

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((4, 4))
    ax.yaxis.set_major_formatter(fmt)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_correlation_tail(
    spike_data, subset_size=500, corr_window=0.05, corr_step=0.025, seed=0
):
    """Plot correlation tail diagnostics: same-cluster vs different-cluster pairs.

    Args:
        spike_data: SpikeData object
        subset_size: Number of neurons to sample
        corr_window: Correlation window in seconds
        corr_step: Correlation step in seconds
        seed: Random seed

    Returns:
        neuron_ids: The sampled neuron indices used for the analysis
    """
    results, neuron_ids = analyze_correlation_tail(
        spike_data,
        subset_size=subset_size,
        corr_window=corr_window,
        corr_step=corr_step,
        seed=seed,
    )

    cluster_size = spike_data.cluster_params.cluster_size
    n_e = spike_data.model_params.N_E

    bins = np.linspace(-0.5, 1.0, 80)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].hist(
        results["Uniform"]["all"],
        bins=bins,
        histtype="step",
        linewidth=2,
        color="black",
        density=True,
        label="All pairs",
    )
    axes[0].hist(
        results["Uniform"]["same"],
        bins=bins,
        histtype="step",
        linewidth=2,
        color="gray",
        density=True,
        label="Same-cluster pairs",
    )
    axes[0].hist(
        results["Uniform"]["diff"],
        bins=bins,
        histtype="step",
        linewidth=2,
        color="dodgerblue",
        density=True,
        label="Different-cluster pairs",
    )
    axes[0].set_title("Uniform network")
    axes[0].set_xlabel("Correlation coefficient")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(
        results["Clustered"]["all"],
        bins=bins,
        histtype="step",
        linewidth=2,
        color="limegreen",
        density=True,
        label="All pairs",
    )
    axes[1].hist(
        results["Clustered"]["same"],
        bins=bins,
        histtype="step",
        linewidth=2,
        color="orange",
        density=True,
        label="Same-cluster pairs",
    )
    axes[1].hist(
        results["Clustered"]["diff"],
        bins=bins,
        histtype="step",
        linewidth=2,
        color="dodgerblue",
        density=True,
        label="Different-cluster pairs",
    )
    axes[1].set_title("Clustered network")
    axes[1].set_xlabel("Correlation coefficient")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    def brief_stats(x):
        if x.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        return np.mean(x), np.quantile(x, 0.95), np.quantile(x, 0.99), np.mean(x > 0.2)

    print("")
    print("Summary (mean, q95, q99, p[r>0.2]):")
    for network in ["Uniform", "Clustered"]:
        for pair_type in ["all", "same", "diff"]:
            m, q95, q99, p02 = brief_stats(results[network][pair_type])
            print(
                f"{network:9s} {pair_type:4s} | "
                f"{m: .4f}, {q95: .3f}, {q99: .3f}, {p02: .3f}"
            )

    n_pairs_total = n_e * (n_e - 1) // 2
    n_clusters = n_e // cluster_size
    n_same_pairs = n_clusters * (cluster_size * (cluster_size - 1) // 2)
    print("")
    print(
        f"Same-cluster pair opportunities in full network: "
        f"{n_same_pairs:,}/{n_pairs_total:,} ({n_same_pairs / n_pairs_total:.2%})"
    )

    print("")
    print("Tail composition in clustered network:")
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        all_count = np.sum(results["Clustered"]["all"] > threshold)
        same_count = np.sum(results["Clustered"]["same"] > threshold)
        frac_same = same_count / all_count if all_count > 0 else np.nan
        print(
            f"  r > {threshold:.1f}: same-cluster share = "
            f"{frac_same:.1%} ({same_count:,}/{all_count:,})"
        )

    return neuron_ids
