import numpy as np


def _count_spikes_in_window(spike_times, spike_ids, n_neurons, start_t, end_t):
    """Count spikes per neuron in a time window."""
    mask = (spike_times >= start_t) & (spike_times < end_t)
    return np.bincount(spike_ids[mask], minlength=n_neurons)


def compute_firing_rates(spike_data, window_t=None):
    """Compute firing rates from spike data.

    Args:
        spike_data: SpikeData object
        window_t: Analysis window duration in seconds (float).
                  Default: model_params.firing_rate_window_t

    Returns:
        (uniform_rates, clustered_rates): Flattened arrays of firing rates
    """
    params = spike_data.model_params
    start_t = float(params.analysis_start_t)
    window_t = float(window_t or params.firing_rate_window_t)
    end_t = start_t + window_t
    n_e = params.N_E

    def compute_rates(spikes_list):
        rates = np.zeros(
            (spike_data.realizations, spike_data.trials, n_e), dtype=np.float64
        )
        for r, trial_spikes in enumerate(spikes_list):
            for t, (times, ids) in enumerate(trial_spikes):
                counts = _count_spikes_in_window(times, ids, n_e, start_t, end_t)
                rates[r, t, :] = counts / window_t
        return rates.ravel()

    return compute_rates(spike_data.uniform), compute_rates(spike_data.clustered)
