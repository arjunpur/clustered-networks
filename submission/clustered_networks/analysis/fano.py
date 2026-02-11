import numpy as np

from .firing_rate import _count_spikes_in_window


def compute_fano_factor(spike_data, window_t=None):
    """Compute Fano factor from spike data.

    Args:
        spike_data: SpikeData object
        window_t: Analysis window duration in seconds (float).
                  Default: model_params.fano_factor_window_t

    Returns:
        (uniform_ff, clustered_ff): Fano factors of shape (realizations, N_E)
    """
    params = spike_data.model_params
    start_t = float(params.analysis_start_t)
    window_t = float(window_t or params.fano_factor_window_t)
    end_t = start_t + window_t
    n_e = params.N_E

    def compute_ff(spikes_list):
        counts = np.zeros(
            (spike_data.realizations, spike_data.trials, n_e), dtype=np.int32
        )
        for r, trial_spikes in enumerate(spikes_list):
            for t, (times, ids) in enumerate(trial_spikes):
                counts[r, t, :] = _count_spikes_in_window(
                    times, ids, n_e, start_t, end_t
                )

        mean = counts.mean(axis=1)
        var = counts.var(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(mean > 0, var / mean, np.nan)

    return compute_ff(spike_data.uniform), compute_ff(spike_data.clustered)


def compute_fano_factor_for_ree(R_ee_values, n_trials=3, duration_s=3.0, seed=42):
    """Compute mean Fano Factor for networks with different R_ee values.

    This function requires Brian2 (imports NeuronNetwork) because it runs
    simulations for each R_ee value.

    Args:
        R_ee_values: Array of R_ee values to test
        n_trials: Number of trials per R_ee value
        duration_s: Simulation duration in seconds
        seed: Random seed

    Returns:
        Array of mean Fano factors, one per R_ee value
    """
    from brian2 import second

    from ..experiment import SpikeData
    from ..network import NeuronNetwork
    from ..params import ClusterParams, ModelParams

    mean_fano_factors = []
    model_params = ModelParams(duration=duration_s * second)

    for R_ee in R_ee_values:
        print(f"Running R_ee = {R_ee}...")

        cluster_params = ClusterParams(enabled=True, R_ee=R_ee)
        network = NeuronNetwork(model_params, cluster_params, seed=seed)

        # Collect spike data from trials
        trial_spikes = []
        for t in range(n_trials):
            network.run()
            spike_times = np.array(network.spike_monitor_e.t)
            spike_ids = np.array(network.spike_monitor_e.i)
            trial_spikes.append((spike_times, spike_ids))

        # Create SpikeData with only clustered data
        spike_data = SpikeData(
            uniform=[],  # Empty - not needed
            clustered=[trial_spikes],  # 1 realization
            model_params=model_params,
            cluster_params=cluster_params,
            realizations=1,
            trials=n_trials,
        )

        # Compute Fano factor (only use clustered result)
        _, clustered_ff = compute_fano_factor(spike_data)

        mean_ff = np.nanmean(clustered_ff)
        mean_fano_factors.append(mean_ff)
        print(f"  Mean Fano Factor: {mean_ff:.3f}")

    return np.array(mean_fano_factors)
