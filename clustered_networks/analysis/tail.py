import numpy as np


def _window_counts_subset(
    spike_times, spike_ids, n_e, neuron_ids, window, step, timerange
):
    start_t, end_t = float(timerange[0]), float(timerange[1])
    window = float(window)
    step = float(step)
    starts = np.arange(start_t, end_t - window + step, step)

    counts = np.zeros((len(neuron_ids), len(starts)), dtype=np.float64)
    spike_times = np.asarray(spike_times)
    spike_ids = np.asarray(spike_ids, dtype=np.int32)

    interval = (spike_times >= start_t) & (spike_times <= end_t)
    spike_times = spike_times[interval]
    spike_ids = spike_ids[interval]

    for w, ws in enumerate(starts):
        mask = (spike_times >= ws) & (spike_times <= ws + window)
        if np.any(mask):
            b = np.bincount(spike_ids[mask], minlength=n_e)
            counts[:, w] = b[neuron_ids]

    return counts


def analyze_correlation_tail(
    spike_data, subset_size=500, corr_window=0.05, corr_step=0.025, seed=0
):
    """Compute correlation tail statistics for uniform vs clustered networks.

    Splits correlations into same-cluster and different-cluster pairs to
    explain the long right tail in clustered networks.

    Args:
        spike_data: SpikeData object
        subset_size: Number of neurons to sample for analysis
        corr_window: Correlation window in seconds (default 0.05 = 50ms)
        corr_step: Correlation step in seconds (default 0.025 = 25ms)
        seed: Random seed

    Returns:
        (results, neuron_ids): results is a dict with keys "Uniform" and
        "Clustered", each containing "all", "same", "diff" arrays of
        correlation coefficients. neuron_ids is the sampled neuron indices.
    """
    params = spike_data.model_params
    cluster_size = spike_data.cluster_params.cluster_size
    n_e = params.N_E
    start_t = float(params.analysis_start_t)
    end_t = start_t + float(params.analysis_window_t)
    timerange = (start_t, end_t)

    rng = np.random.default_rng(seed)
    subset_size = min(subset_size, n_e)
    neuron_ids = np.sort(rng.choice(n_e, size=subset_size, replace=False))

    cluster_ids = neuron_ids // cluster_size
    pair_i, pair_j = np.triu_indices(subset_size, k=1)
    same_mask = cluster_ids[pair_i] == cluster_ids[pair_j]

    results = {}
    for name, spikes_list in [
        ("Uniform", spike_data.uniform),
        ("Clustered", spike_data.clustered),
    ]:
        all_vals, same_vals, diff_vals = [], [], []

        for trial_spikes in spikes_list:  # one realization
            cov_sum = np.zeros((subset_size, subset_size), dtype=np.float64)
            var_sum = np.zeros(subset_size, dtype=np.float64)
            spike_count_sum = np.zeros(subset_size, dtype=np.int64)
            n_trials = len(trial_spikes)

            for times, ids in trial_spikes:
                times = np.asarray(times)
                ids = np.asarray(ids, dtype=np.int32)

                interval_mask = (times >= start_t) & (times <= end_t)
                if np.any(interval_mask):
                    b = np.bincount(ids[interval_mask], minlength=n_e)
                    spike_count_sum += b[neuron_ids]

                x = _window_counts_subset(
                    times, ids, n_e, neuron_ids, corr_window, corr_step, timerange
                )
                cov_sum += np.cov(x, rowvar=True, bias=False)
                var_sum += np.var(x, axis=1, ddof=1)

            mean_cov = cov_sum / n_trials
            mean_var = var_sum / n_trials
            active_local = spike_count_sum > 0

            denom = np.sqrt(mean_var[pair_i] * mean_var[pair_j])
            valid = (denom > 0) & active_local[pair_i] & active_local[pair_j]

            vals = mean_cov[pair_i[valid], pair_j[valid]] / denom[valid]
            same_valid = same_mask[valid]

            all_vals.append(vals)
            same_vals.append(vals[same_valid])
            diff_vals.append(vals[~same_valid])

        results[name] = {
            "all": (
                np.concatenate(all_vals) if all_vals else np.array([], dtype=np.float64)
            ),
            "same": (
                np.concatenate(same_vals)
                if same_vals
                else np.array([], dtype=np.float64)
            ),
            "diff": (
                np.concatenate(diff_vals)
                if diff_vals
                else np.array([], dtype=np.float64)
            ),
        }

    return results, neuron_ids
