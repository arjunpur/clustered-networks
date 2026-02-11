import numpy as np


def _window_spikes_all_neurons(
    spike_times, spike_ids, n_neurons, window, step, timerange
):
    """Compute spike counts in sliding windows for all neurons."""
    start_time, end_time = float(timerange[0]), float(timerange[1])
    window, step = float(window), float(step)

    window_starts = np.arange(start_time, end_time - window + step, step)
    n_windows = len(window_starts)

    window_counts = np.zeros((n_neurons, n_windows), dtype=np.int32)
    spike_times = np.asarray(spike_times)
    spike_ids = np.asarray(spike_ids)

    for w, ws in enumerate(window_starts):
        mask = (spike_times >= ws) & (spike_times <= ws + window)
        window_counts[:, w] = np.bincount(spike_ids[mask], minlength=n_neurons)

    return window_counts


def _compute_realization_correlation_data(
    trial_spikes,
    n_e,
    corr_window,
    corr_step,
    timerange,
    start_t_float,
    end_t_float,
):
    """Return trial-averaged covariance/variance and active-neuron mask for one realization."""
    n_trials = len(trial_spikes)
    if n_trials == 0:
        return None, None, None

    cov_sum = np.zeros((n_e, n_e), dtype=np.float64)
    var_sum = np.zeros(n_e, dtype=np.float64)
    spike_count_sum = np.zeros(n_e, dtype=np.int64)

    for times, ids in trial_spikes:
        times = np.asarray(times)
        ids = np.asarray(ids, dtype=np.int32)

        analysis_mask = (times >= start_t_float) & (times <= end_t_float)
        if np.any(analysis_mask):
            spike_count_sum += np.bincount(ids[analysis_mask], minlength=n_e)

        window_counts = _window_spikes_all_neurons(
            times, ids, n_e, corr_window, corr_step, timerange
        )
        cov_sum += np.cov(window_counts, rowvar=True, bias=False)
        var_sum += np.var(window_counts, axis=1, ddof=1)

    mean_cov = cov_sum / n_trials
    mean_var = var_sum / n_trials
    active_neurons = spike_count_sum > 0
    return mean_cov, mean_var, active_neurons


def _extract_correlation_coefficients_from_realization(
    mean_cov,
    mean_var,
    active_neurons,
    n_e,
    same_cluster_only=False,
    cluster_size=None,
    return_pairs=False,
):
    """Extract correlation coefficients for valid active-neuron pairs from one realization."""
    coeffs = []
    pair_i = []
    pair_j = []

    active_idx = np.flatnonzero(active_neurons)
    if len(active_idx) < 2:
        if return_pairs:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )
        return np.array([], dtype=np.float64)

    if same_cluster_only and cluster_size is not None:
        for idx_i, i in enumerate(active_idx[:-1]):
            cluster_end = min((i // cluster_size + 1) * cluster_size, n_e)
            for j in active_idx[idx_i + 1 :]:
                if j >= cluster_end:
                    break
                denom = np.sqrt(mean_var[i] * mean_var[j])
                if denom > 0:
                    coeffs.append(mean_cov[i, j] / denom)
                    if return_pairs:
                        pair_i.append(i)
                        pair_j.append(j)
    else:
        for idx_i, i in enumerate(active_idx[:-1]):
            for j in active_idx[idx_i + 1 :]:
                denom = np.sqrt(mean_var[i] * mean_var[j])
                if denom > 0:
                    coeffs.append(mean_cov[i, j] / denom)
                    if return_pairs:
                        pair_i.append(i)
                        pair_j.append(j)

    coeffs = np.array(coeffs, dtype=np.float64)
    if return_pairs:
        return (
            coeffs,
            np.array(pair_i, dtype=np.int32),
            np.array(pair_j, dtype=np.int32),
        )
    return coeffs


def _get_realization_pair_coefficients(
    trial_spikes,
    n_e,
    corr_window,
    corr_step,
    timerange,
    start_t_float,
    end_t_float,
    same_cluster_only=False,
    cluster_size=None,
    return_pairs=False,
):
    """Shared realization-level correlation extraction used by plotting and analysis."""
    mean_cov, mean_var, active_neurons = _compute_realization_correlation_data(
        trial_spikes,
        n_e,
        corr_window,
        corr_step,
        timerange,
        start_t_float,
        end_t_float,
    )

    if mean_cov is None:
        if return_pairs:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )
        return np.array([], dtype=np.float64)

    return _extract_correlation_coefficients_from_realization(
        mean_cov,
        mean_var,
        active_neurons,
        n_e,
        same_cluster_only=same_cluster_only,
        cluster_size=cluster_size,
        return_pairs=return_pairs,
    )


def compute_correlation_coefficients(
    spike_data, corr_window=0.05, corr_step=0.025, same_cluster_only=False
):
    """Compute correlation coefficients from spike data.

    Args:
        spike_data: SpikeData object
        corr_window: Sliding window size in seconds (default 0.05 = 50ms)
        corr_step: Step size in seconds (default 0.025 = 25ms)
        same_cluster_only: If True, only compute for same-cluster pairs

    Returns:
        (uniform_coeffs, clustered_coeffs): Arrays of correlation coefficients
    """
    params = spike_data.model_params
    cluster_params = spike_data.cluster_params
    start_t = float(params.analysis_start_t)
    end_t = start_t + float(params.analysis_window_t)
    timerange = (start_t, end_t)
    n_e = params.N_E

    use_same_cluster = same_cluster_only and cluster_params.enabled
    cluster_size = cluster_params.cluster_size if use_same_cluster else None

    def compute_coeffs_by_realization(spikes_list):
        realization_coeffs = []
        for trial_spikes in spikes_list:
            coeffs = _get_realization_pair_coefficients(
                trial_spikes,
                n_e,
                corr_window,
                corr_step,
                timerange,
                start_t,
                end_t,
                same_cluster_only=use_same_cluster,
                cluster_size=cluster_size,
                return_pairs=False,
            )
            if coeffs.size > 0:
                realization_coeffs.append(coeffs)

        if realization_coeffs:
            return np.concatenate(realization_coeffs)
        return np.array([], dtype=np.float64)

    return (
        compute_coeffs_by_realization(spike_data.uniform),
        compute_coeffs_by_realization(spike_data.clustered),
    )
