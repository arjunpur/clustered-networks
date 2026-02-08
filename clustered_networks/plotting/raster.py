import numpy as np
import matplotlib.pyplot as plt

from ..analysis.correlation import (
    _window_spikes_all_neurons,
    _get_realization_pair_coefficients,
)


def plot_spike_raster(
    spike_monitor_e,
    spike_monitor_i=None,
    n_e=0,
    title="Spike raster",
    ax=None,
    color="gray",
    alpha=0.5,
):
    """Plot a spike raster for excitatory (and optionally inhibitory) neurons.

    Args:
        spike_monitor_e: Brian2 SpikeMonitor for excitatory neurons
        spike_monitor_i: Brian2 SpikeMonitor for inhibitory neurons (optional)
        n_e: Number of excitatory neurons (used to offset inhibitory neuron indices)
        title: Plot title
        ax: Matplotlib axes to plot on. If None, creates a new figure.
        color: Color for the dots (default: 'gray')
        alpha: Transparency for the dots (default: 0.5)

    Returns:
        The axes object
    """
    from brian2 import ms

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        spike_monitor_e.t / ms,
        spike_monitor_e.i,
        ".",
        markersize=2,
        color=color,
        alpha=alpha,
        label="E",
    )
    if spike_monitor_i is not None:
        ax.plot(
            spike_monitor_i.t / ms,
            spike_monitor_i.i + n_e,
            ".",
            markersize=2,
            color=color,
            alpha=alpha,
            label="I",
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return ax


def plot_trial_rasters(
    network,
    n_trials=9,
    ncols=3,
    figsize=None,
    show_inhibitory=False,
    save_path=None,
    color="gray",
    alpha=0.5,
):
    """Plot spike rasters for multiple trials of a network in a grid.

    Args:
        network: NeuronNetwork instance
        n_trials: Number of trials to plot
        ncols: Number of columns in the grid
        figsize: Figure size (auto-calculated if None)
        show_inhibitory: Whether to show inhibitory neurons
        save_path: Path to save the figure (optional)
        color: Color for the dots (default: 'gray')
        alpha: Transparency for the dots (default: 0.5)

    Returns:
        (fig, axes): Matplotlib figure and axes
    """
    nrows = int(np.ceil(n_trials / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)  # Ensure 2D array even if nrows=1

    for trial in range(n_trials):
        row = trial // ncols
        col = trial % ncols
        ax = axes[row, col]

        network.run()

        spike_monitor_i = network.spike_monitor_i if show_inhibitory else None
        n_e = network.params.N_E if show_inhibitory else 0

        plot_spike_raster(
            network.spike_monitor_e,
            spike_monitor_i,
            n_e=n_e,
            title=f"Trial {trial + 1}",
            ax=ax,
            color=color,
            alpha=alpha,
        )

        # Only show legend on first plot to reduce clutter
        if trial > 0:
            ax.get_legend().remove()

    # Hide any unused subplots
    for idx in range(n_trials, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_rasters_from_spike_data(
    spike_data,
    n_trials=4,
    ncols=2,
    network_type="clustered",
    realization=0,
    color="gray",
    alpha=0.5,
    neuron_fraction=0.5,
    seed=42,
    save_path=None,
):
    """Plot spike rasters from stored SpikeData.

    Args:
        spike_data: SpikeData object containing spike times and IDs
        n_trials: Number of trials to plot
        ncols: Number of columns in the grid
        network_type: 'uniform' or 'clustered'
        realization: Which realization to plot (default: 0)
        color: Color for the dots
        alpha: Transparency for the dots
        neuron_fraction: Fraction of neurons to plot (default: 0.5)
        seed: Random seed for neuron selection
        save_path: Path to save the figure

    Returns:
        (fig, axes): Matplotlib figure and axes
    """
    from brian2 import ms

    spikes_list = (
        spike_data.clustered if network_type == "clustered" else spike_data.uniform
    )
    trial_spikes = spikes_list[realization]

    n_trials = min(n_trials, len(trial_spikes))
    nrows = int(np.ceil(n_trials / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    # Get duration from model params for x-axis
    duration_ms = float(spike_data.model_params.duration / ms)
    n_e = spike_data.model_params.N_E

    # Select a random subset of neurons to plot
    rng = np.random.default_rng(seed)
    n_neurons_to_plot = int(n_e * neuron_fraction)
    selected_neurons = set(rng.choice(n_e, size=n_neurons_to_plot, replace=False))

    for trial in range(n_trials):
        row = trial // ncols
        col = trial % ncols
        ax = axes[row, col]

        spike_times, spike_ids = trial_spikes[trial]
        # Convert to ms if needed (spike_times are in seconds from brian2)
        times_ms = spike_times * 1000 if spike_times.max() < 100 else spike_times

        # Filter to only selected neurons
        mask = np.array([sid in selected_neurons for sid in spike_ids])
        times_ms = times_ms[mask]
        spike_ids = spike_ids[mask]

        ax.plot(times_ms, spike_ids, ".", markersize=2, color=color, alpha=alpha)
        ax.set_title(f"Trial {trial + 1}")
        ax.set_ylabel("Neuron index")
        ax.set_xlabel("Time (ms)")
        ax.set_xlim(0, duration_ms)
        ax.set_ylim(0, n_e)

    # Hide any unused subplots
    for idx in range(n_trials, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    title = (
        f"{'Clustered' if network_type == 'clustered' else 'Uniform'} Network "
        f"- Realization {realization + 1}"
    )
    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_high_corr_pair_rasters(
    spike_data,
    network_type="clustered",
    realization=0,
    trial=0,
    corr_window=0.05,
    corr_step=0.025,
    same_cluster_only=False,
    n_pairs=6,
    threshold=0.4,
    seed=1,
):
    """Plot rasters for a sample of high-correlation neuron pairs.

    Args:
        spike_data: SpikeData object
        network_type: 'uniform' or 'clustered'
        realization: Which realization to use
        trial: Which trial to plot
        corr_window: Correlation window in seconds (default 0.05 = 50ms)
        corr_step: Correlation step in seconds (default 0.025 = 25ms)
        same_cluster_only: If True, only consider same-cluster pairs
        n_pairs: Number of pairs to plot
        threshold: Minimum correlation threshold
        seed: Random seed for pair selection

    Returns:
        (chosen, coeffs, (pair_i, pair_j))
    """
    params = spike_data.model_params
    cluster_params = spike_data.cluster_params

    n_e = params.N_E
    start_t_float = float(params.analysis_start_t)
    end_t_float = float(params.analysis_start_t + params.analysis_window_t)
    timerange = (start_t_float, end_t_float)

    spikes_list = (
        spike_data.clustered if network_type == "clustered" else spike_data.uniform
    )
    trial_spikes_realization = spikes_list[realization]

    use_same_cluster = same_cluster_only and cluster_params.enabled
    cluster_size = cluster_params.cluster_size if use_same_cluster else None

    coeffs, pair_i, pair_j = _get_realization_pair_coefficients(
        trial_spikes_realization,
        n_e,
        corr_window,
        corr_step,
        timerange,
        start_t_float,
        end_t_float,
        same_cluster_only=use_same_cluster,
        cluster_size=cluster_size,
        return_pairs=True,
    )

    if coeffs.size == 0:
        print(
            "No valid pairs found for this realization (after active-neuron filtering)."
        )
        return np.array([], dtype=np.int32), coeffs, (pair_i, pair_j)

    # Trial-specific activity filter for raster plotting
    spike_times_trial, spike_ids_trial = spikes_list[realization][trial]
    spike_times_trial = np.asarray(spike_times_trial, dtype=float)
    spike_ids_trial = np.asarray(spike_ids_trial, dtype=np.int32)

    in_range_trial = (spike_times_trial >= start_t_float) & (
        spike_times_trial <= end_t_float
    )
    spike_times_trial = spike_times_trial[in_range_trial]
    spike_ids_trial = spike_ids_trial[in_range_trial]

    trial_counts = np.bincount(spike_ids_trial, minlength=n_e)
    trial_pair_active = (trial_counts[pair_i] > 0) & (trial_counts[pair_j] > 0)

    active_idx = np.where(trial_pair_active)[0]
    if active_idx.size == 0:
        print(
            "No pairs have spikes from BOTH neurons in the selected trial analysis window."
        )
        return np.array([], dtype=np.int32), coeffs, (pair_i, pair_j)

    rng = np.random.default_rng(seed)
    high_idx = active_idx[coeffs[active_idx] >= threshold]

    if len(high_idx) == 0:
        sorted_active = active_idx[np.argsort(coeffs[active_idx])[::-1]]
        chosen = sorted_active[: min(n_pairs, len(sorted_active))]
        print(
            f"No trial-active pairs found above threshold r >= {threshold:.2f}; "
            f"showing top trial-active pairs instead."
        )
    else:
        n_select = min(n_pairs, len(high_idx))
        chosen = rng.choice(high_idx, size=n_select, replace=False)
        chosen = chosen[np.argsort(coeffs[chosen])[::-1]]
        print(
            f"Found {len(high_idx)} trial-active pairs above threshold "
            f"r >= {threshold:.2f}; plotting {len(chosen)} sampled pairs."
        )

    print(
        f"Filtered out {np.sum(~trial_pair_active)} pairs because at least one "
        f"neuron had 0 spikes in plotted trial window."
    )
    print(
        "Note: displayed r is realization-level (averaged across all trials "
        "in this realization)."
    )
    print(
        "A pair can have high realization-level r even if spikes are "
        "sparse/non-overlapping in the plotted trial."
    )

    realization_counts = np.zeros(n_e, dtype=np.int64)
    for times_r, ids_r in trial_spikes_realization:
        times_r = np.asarray(times_r, dtype=float)
        ids_r = np.asarray(ids_r, dtype=np.int32)
        in_range_r = (times_r >= start_t_float) & (times_r <= end_t_float)
        if np.any(in_range_r):
            realization_counts += np.bincount(ids_r[in_range_r], minlength=n_e)

    trial_window_counts = _window_spikes_all_neurons(
        spike_times_trial,
        spike_ids_trial,
        n_e,
        corr_window,
        corr_step,
        timerange,
    )

    fig, axes = plt.subplots(
        len(chosen), 1, figsize=(12, 1.8 * len(chosen)), sharex=True
    )
    if len(chosen) == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen):
        neuron_i, neuron_j = int(pair_i[idx]), int(pair_j[idx])
        r_ij = coeffs[idx]
        count_i = int(trial_counts[neuron_i])
        count_j = int(trial_counts[neuron_j])
        total_count_i = int(realization_counts[neuron_i])
        total_count_j = int(realization_counts[neuron_j])
        x_i_trial = trial_window_counts[neuron_i].astype(np.float64)
        x_j_trial = trial_window_counts[neuron_j].astype(np.float64)
        var_i_trial = np.var(x_i_trial, ddof=1)
        var_j_trial = np.var(x_j_trial, ddof=1)
        if var_i_trial > 0 and var_j_trial > 0:
            cov_trial = np.cov(x_i_trial, x_j_trial, bias=False)[0, 1]
            r_trial = cov_trial / np.sqrt(var_i_trial * var_j_trial)
            r_trial_str = f"{r_trial:.3f}"
        else:
            r_trial_str = "nan"

        t_i = spike_times_trial[spike_ids_trial == neuron_i]
        t_j = spike_times_trial[spike_ids_trial == neuron_j]

        ax.vlines(t_i, 0.7, 1.3, color="tab:blue", linewidth=0.8)
        ax.vlines(t_j, 1.7, 2.3, color="tab:orange", linewidth=0.8)
        ax.set_yticks([1, 2])
        ax.set_yticklabels([f"n={neuron_i}", f"n={neuron_j}"])
        ax.set_ylim(0.4, 2.6)
        ax.set_title(
            f"Pair ({neuron_i}, {neuron_j}), r_real={r_ij:.3f}, "
            f"r_trial={r_trial_str}, "
            f"trial_spikes=({count_i}, {count_j}), "
            f"realization_spikes=({total_count_i}, {total_count_j}) | "
            f"{network_type}, realization={realization}, trial={trial}, "
            f"same_cluster_only={use_same_cluster}"
        )

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    return chosen, coeffs, (pair_i, pair_j)


def plot_stim_raster(stim):
    """Plot stimulus raster with stimulus indicator bar.

    Args:
        stim: Tuple (start, end) or list of tuples for stimulus intervals.
              Times should be Brian2 quantities.

    Returns:
        (ax_raster, ax_stim): Tuple of axes for raster and stimulus panels
    """
    from brian2 import ms

    fig, (ax_r, ax_s) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )

    ax_r.tick_params(labelbottom=False)
    intervals = [stim] if isinstance(stim, tuple) else stim
    for t0, t1 in intervals:
        ax_s.axvspan(t0 / ms, t1 / ms, color="k", alpha=1)
    ax_s.set_ylabel("Stim")
    ax_s.set_yticks([])
    ax_s.set_xlabel("Time (ms)")
    return (ax_r, ax_s)
