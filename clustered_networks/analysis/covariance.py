import numpy as np


def _validate_max_lag(max_lag_bins, n_bins):
    if max_lag_bins < 0:
        raise ValueError(f"max_lag_bins must be >= 0, got {max_lag_bins}.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")
    if max_lag_bins >= n_bins:
        raise ValueError(
            "max_lag_ms is too large for the chosen window/bin size; "
            f"need max_lag_bins < n_bins, got {max_lag_bins} >= {n_bins}."
        )


def _compute_bin_config(bin_size_ms, time_range_s, max_lag_ms):
    if bin_size_ms <= 0:
        raise ValueError(f"bin_size_ms must be > 0, got {bin_size_ms}.")
    if max_lag_ms < 0:
        raise ValueError(f"max_lag_ms must be >= 0, got {max_lag_ms}.")

    t0, t1 = map(float, time_range_s)
    if t1 <= t0:
        raise ValueError(f"time_range_s must satisfy end > start, got ({t0}, {t1}).")

    bin_size_s = bin_size_ms / 1000.0
    n_bins = int(np.floor((t1 - t0) / bin_size_s))
    if n_bins < 1:
        raise ValueError(
            "Window/binning produced zero bins; increase time_range_s or decrease bin_size_ms."
        )

    max_lag_bins = int(max_lag_ms / bin_size_ms)
    _validate_max_lag(max_lag_bins, n_bins)

    lags_ms = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size_ms
    return t0, t1, bin_size_s, n_bins, max_lag_bins, lags_ms


def _coerce_spike_arrays(times_s, ids):
    times_s = np.asarray(times_s, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if times_s.shape[0] != ids.shape[0]:
        raise ValueError(
            f"times_s and ids must have same length, got {times_s.shape[0]} and {ids.shape[0]}."
        )
    return times_s, ids


def bin_spikes_to_counts(times_s, ids, n_neurons, bin_size_s, t_start_s, t_end_s):
    """
    Return spike COUNTS matrix of shape (n_neurons, n_bins) for [t_start_s, t_end_s).
    times_s: 1D array of spike times in seconds
    ids:     1D array of neuron indices (same length as times_s)
    """
    if n_neurons <= 0:
        raise ValueError(f"n_neurons must be > 0, got {n_neurons}.")
    if bin_size_s <= 0:
        raise ValueError(f"bin_size_s must be > 0, got {bin_size_s}.")
    if t_end_s <= t_start_s:
        raise ValueError(
            f"time window must satisfy t_end_s > t_start_s, got ({t_start_s}, {t_end_s})."
        )

    n_bins = int(np.floor((t_end_s - t_start_s) / bin_size_s))
    if n_bins < 1:
        raise ValueError(
            "Window/binning produced zero bins; increase time_range_s or decrease bin_size_ms."
        )

    counts = np.zeros((n_neurons, n_bins), dtype=np.int32)
    times_s, ids = _coerce_spike_arrays(times_s, ids)

    mask = (times_s >= t_start_s) & (times_s < t_end_s)
    if not np.any(mask):
        return counts

    times = times_s[mask]
    neurons = ids[mask]

    if np.any(neurons < 0) or np.any(neurons >= n_neurons):
        raise ValueError(
            f"ids out of range for n_neurons={n_neurons}: "
            f"min={int(neurons.min())}, max={int(neurons.max())}."
        )

    bin_idx = ((times - t_start_s) / bin_size_s).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    np.add.at(counts, (neurons, bin_idx), 1)
    return counts


def counts_to_centered_rates(counts, bin_size_s):
    """
    counts: (n_neurons, n_bins)
    returns centered rate matrix X in Hz: X[i,t] = rate(i,t) - mean_rate(i)
    """
    if bin_size_s <= 0:
        raise ValueError(f"bin_size_s must be > 0, got {bin_size_s}.")

    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D, got shape {counts.shape}.")

    rates = counts.astype(np.float64, copy=False) / bin_size_s
    rates -= rates.mean(axis=1, keepdims=True)
    return rates


# ----------------------------
# Covariance curves (fast, clear)
# ----------------------------


def mean_autocov_curve(X, max_lag_bins):
    """
    Mean spike-train autocovariance across neurons as a function of lag.
    X is centered per neuron, shape (n_neurons, n_bins), units Hz.
    Returns array length 2*max_lag_bins+1, lags [-max..+max].
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")

    _, n_bins = X.shape
    _validate_max_lag(max_lag_bins, n_bins)

    out = np.empty(2 * max_lag_bins + 1, dtype=np.float64)
    mid = max_lag_bins
    out[mid] = np.mean(X * X)

    for lag in range(1, max_lag_bins + 1):
        v = np.mean(X[:, :-lag] * X[:, lag:])
        out[mid + lag] = v
        out[mid - lag] = v

    return out


def mean_crosscov_curve(X, pairs, max_lag_bins):
    """
    Mean cross-covariance across a set of pairs as a function of lag.
    X is centered per neuron, shape (n_neurons, n_bins).
    pairs: array shape (P, 2) with i<j indices.
    Returns array length 2*max_lag_bins+1, lags [-max..+max].

    Convention:
      +lag uses X[i,t] * X[j,t+lag] averaged over t and pairs.
      -lag uses X[i,t+lag] * X[j,t].
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")

    n_neurons, n_bins = X.shape
    _validate_max_lag(max_lag_bins, n_bins)

    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (P, 2), got {pairs.shape}.")
    if pairs.shape[0] == 0:
        raise ValueError(
            "pairs is empty; need at least one neuron pair to compute cross-covariance."
        )

    i = pairs[:, 0]
    j = pairs[:, 1]
    if (
        np.any(i < 0)
        or np.any(i >= n_neurons)
        or np.any(j < 0)
        or np.any(j >= n_neurons)
    ):
        raise ValueError("pairs contains neuron indices outside [0, n_neurons).")

    out = np.empty(2 * max_lag_bins + 1, dtype=np.float64)
    mid = max_lag_bins
    out[mid] = np.mean(X[i, :] * X[j, :])

    for lag in range(1, max_lag_bins + 1):
        out[mid + lag] = np.mean(X[i, :-lag] * X[j, lag:])
        out[mid - lag] = np.mean(X[i, lag:] * X[j, :-lag])

    return out


# ----------------------------
# Pair sampling
# ----------------------------


def _effective_n_pairs(n_pairs, n_available):
    if n_pairs < 0:
        raise ValueError(f"n_pairs must be >= 0, got {n_pairs}.")
    return min(n_pairs, n_available)


def sample_pairs_uniform(n_neurons, n_pairs, rng):
    """Sample up to n_pairs unique unordered pairs (i<j) uniformly."""
    if n_neurons < 2:
        raise ValueError(f"n_neurons must be >= 2, got {n_neurons}.")

    n_available = n_neurons * (n_neurons - 1) // 2
    n_pairs_eff = _effective_n_pairs(n_pairs, n_available)
    if n_pairs_eff == 0:
        return np.empty((0, 2), dtype=np.int64)

    # Dense request: sample from full pair table exactly.
    if n_pairs_eff > n_available // 4:
        ii, jj = np.triu_indices(n_neurons, k=1)
        all_pairs = np.column_stack((ii, jj)).astype(np.int64, copy=False)
        if n_pairs_eff == n_available:
            return all_pairs
        chosen = rng.choice(n_available, size=n_pairs_eff, replace=False)
        return all_pairs[chosen]

    # Sparse request: rejection sampling avoids large pair tables.
    pairs = set()
    while len(pairs) < n_pairs_eff:
        a = int(rng.integers(0, n_neurons))
        b = int(rng.integers(0, n_neurons - 1))
        if b >= a:
            b += 1
        i, j = (a, b) if a < b else (b, a)
        pairs.add((i, j))

    return np.array(list(pairs), dtype=np.int64)


def sample_pairs_within_clusters(cluster_ids, n_pairs, rng):
    """
    Sample up to n_pairs unique unordered pairs (i<j) uniformly from within-cluster pairs.
    Works for equal or unequal cluster sizes.
    """
    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.ndim != 1:
        raise ValueError(f"cluster_ids must be 1D, got shape {cluster_ids.shape}.")

    clusters = {}
    for idx, cid in enumerate(cluster_ids):
        clusters.setdefault(int(cid), []).append(idx)

    members = [np.array(v, dtype=np.int64) for v in clusters.values() if len(v) >= 2]
    if not members:
        raise ValueError(
            "No clusters with >=2 neurons; cannot sample within-cluster pairs."
        )

    pair_blocks = []
    for m in members:
        ii, jj = np.triu_indices(len(m), k=1)
        pair_blocks.append(np.column_stack((m[ii], m[jj])))

    all_pairs = np.vstack(pair_blocks).astype(np.int64, copy=False)
    n_pairs_eff = _effective_n_pairs(n_pairs, all_pairs.shape[0])
    if n_pairs_eff == 0:
        return np.empty((0, 2), dtype=np.int64)
    if n_pairs_eff == all_pairs.shape[0]:
        return all_pairs

    chosen = rng.choice(all_pairs.shape[0], size=n_pairs_eff, replace=False)
    return all_pairs[chosen]


def infer_cluster_ids(n_neurons, cluster_size):
    """Infer per-neuron cluster ids with the floor-division scheme used in network construction."""
    if n_neurons <= 0:
        raise ValueError(f"n_neurons must be > 0, got {n_neurons}.")
    if cluster_size <= 0:
        raise ValueError(f"cluster_size must be > 0, got {cluster_size}.")
    return np.arange(n_neurons, dtype=np.int64) // int(cluster_size)


# ----------------------------
# Full analysis: average over trials x realizations
# ----------------------------


def compute_covariance_condition(
    spikes_by_realization,
    n_neurons,
    bin_size_ms=2,
    time_range_s=(1.5, 3.0),
    max_lag_ms=200,
    within_cluster_pairs=False,
    cluster_ids=None,
    max_pairs=5000,
    seed=0,
    normalize_by_mean_rate=False,
):
    """
    spikes_by_realization: list of realizations; each realization is a list of trials;
                          each trial is (times_s, ids) arrays.
    Returns: lags_ms, mean_auto, mean_cross
    """
    if n_neurons < 2:
        raise ValueError(f"n_neurons must be >= 2, got {n_neurons}.")
    if max_pairs <= 0:
        raise ValueError(f"max_pairs must be > 0, got {max_pairs}.")

    t0, t1, bin_size_s, _, max_lag_bins, lags_ms = _compute_bin_config(
        bin_size_ms=bin_size_ms,
        time_range_s=time_range_s,
        max_lag_ms=max_lag_ms,
    )

    spikes_by_realization = list(spikes_by_realization)
    if len(spikes_by_realization) == 0:
        raise ValueError("spikes_by_realization is empty.")

    if within_cluster_pairs:
        if cluster_ids is None:
            raise ValueError("within_cluster_pairs=True requires cluster_ids.")
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.shape[0] != n_neurons:
            raise ValueError(
                f"cluster_ids length must match n_neurons ({n_neurons}), "
                f"got {cluster_ids.shape[0]}."
            )

    auto_sum = np.zeros(len(lags_ms), dtype=np.float64)
    cross_sum = np.zeros(len(lags_ms), dtype=np.float64)
    n_trials_total = 0

    for r, trials in enumerate(spikes_by_realization):
        trials = list(trials)
        if len(trials) == 0:
            continue

        rng = np.random.default_rng(seed + r)
        if within_cluster_pairs:
            pairs = sample_pairs_within_clusters(cluster_ids, max_pairs, rng)
        else:
            pairs = sample_pairs_uniform(n_neurons, max_pairs, rng)

        for times_s, ids in trials:
            counts = bin_spikes_to_counts(times_s, ids, n_neurons, bin_size_s, t0, t1)
            X = counts_to_centered_rates(counts, bin_size_s)

            auto = mean_autocov_curve(X, max_lag_bins)
            cross = mean_crosscov_curve(X, pairs, max_lag_bins)

            if normalize_by_mean_rate:
                mean_rate = counts.sum() / (n_neurons * (t1 - t0))
                if mean_rate > 0:
                    auto /= mean_rate
                    cross /= mean_rate

            auto_sum += auto
            cross_sum += cross
            n_trials_total += 1

    if n_trials_total == 0:
        raise ValueError("No trials found in spikes_by_realization.")

    return lags_ms, auto_sum / n_trials_total, cross_sum / n_trials_total


def compute_covariance_uniform_vs_clustered(
    uniform_spikes,
    clustered_spikes,
    n_neurons,
    cluster_ids=None,
    bin_size_ms=2,
    time_range_s=(1.5, 3.0),
    max_lag_ms=200,
    max_pairs=5000,
    seed=0,
    normalize_by_mean_rate=False,
    clustered_within_cluster_pairs=True,
):
    """
    Returns lags_ms and 4 curves: uniform_auto, uniform_cross, clustered_auto, clustered_cross.
    By default, clustered cross-covariance uses within-cluster pairs (paper Fig 2f logic).
    """
    lags_ms, u_auto, u_cross = compute_covariance_condition(
        uniform_spikes,
        n_neurons=n_neurons,
        bin_size_ms=bin_size_ms,
        time_range_s=time_range_s,
        max_lag_ms=max_lag_ms,
        within_cluster_pairs=False,
        cluster_ids=None,
        max_pairs=max_pairs,
        seed=seed,
        normalize_by_mean_rate=normalize_by_mean_rate,
    )

    _, c_auto, c_cross = compute_covariance_condition(
        clustered_spikes,
        n_neurons=n_neurons,
        bin_size_ms=bin_size_ms,
        time_range_s=time_range_s,
        max_lag_ms=max_lag_ms,
        within_cluster_pairs=clustered_within_cluster_pairs,
        cluster_ids=cluster_ids,
        max_pairs=max_pairs,
        seed=seed + 10_000,
        normalize_by_mean_rate=normalize_by_mean_rate,
    )

    return lags_ms, u_auto, u_cross, c_auto, c_cross


def compute_covariance_uniform_vs_clustered_from_spike_data(
    spike_data,
    bin_size_ms=2,
    time_range_s=None,
    max_lag_ms=200,
    max_pairs=5000,
    seed=0,
    normalize_by_mean_rate=False,
    clustered_within_cluster_pairs=True,
):
    """
    SpikeData-native convenience wrapper.

    Uses:
      - uniform spikes from spike_data.uniform
      - clustered spikes from spike_data.clustered
      - n_neurons from spike_data.model_params.N_E
      - default time_range_s from analysis_start_t to analysis_start_t + analysis_window_t
      - inferred cluster_ids from cluster_size when clustered_within_cluster_pairs=True
    """
    n_neurons = int(spike_data.model_params.N_E)

    if time_range_s is None:
        t0 = float(spike_data.model_params.analysis_start_t)
        t1 = float(
            spike_data.model_params.analysis_start_t
            + spike_data.model_params.analysis_window_t
        )
        time_range_s = (t0, t1)

    use_within_cluster_pairs = bool(
        clustered_within_cluster_pairs
        and getattr(spike_data.cluster_params, "enabled", False)
    )

    cluster_ids = None
    if use_within_cluster_pairs:
        cluster_size = int(spike_data.cluster_params.cluster_size)
        cluster_ids = infer_cluster_ids(n_neurons=n_neurons, cluster_size=cluster_size)

    return compute_covariance_uniform_vs_clustered(
        uniform_spikes=spike_data.uniform,
        clustered_spikes=spike_data.clustered,
        n_neurons=n_neurons,
        cluster_ids=cluster_ids,
        bin_size_ms=bin_size_ms,
        time_range_s=time_range_s,
        max_lag_ms=max_lag_ms,
        max_pairs=max_pairs,
        seed=seed,
        normalize_by_mean_rate=normalize_by_mean_rate,
        clustered_within_cluster_pairs=use_within_cluster_pairs,
    )
