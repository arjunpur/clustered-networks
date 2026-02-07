import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from brian2 import second

from .params import ClusterParams, ModelParams
from .network import NeuronNetwork


@dataclass
class SpikeData:
    """Container for spike data from experiment runs."""

    uniform: list  # [realization][trial] = (spike_times, spike_ids)
    clustered: list  # [realization][trial] = (spike_times, spike_ids)
    model_params: ModelParams
    cluster_params: ClusterParams
    realizations: int
    trials: int

    def save(self, base_dir="data"):
        """Save spike data to disk as .npz files.

        Creates a directory at base_dir/experiment_run_{YYYY-MM-DD_HH-MM-SS}/
        containing one .npz file per network type with all spike data, plus
        a params.json metadata file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(base_dir) / f"experiment_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save spike data for each network type
        for name, spikes_list in [
            ("uniform", self.uniform),
            ("clustered", self.clustered),
        ]:
            arrays = {}
            for r, trial_spikes in enumerate(spikes_list):
                for t, (times, ids) in enumerate(trial_spikes):
                    arrays[f"r{r}_t{t}_times"] = np.asarray(times)
                    arrays[f"r{r}_t{t}_ids"] = np.asarray(ids)
            np.savez_compressed(run_dir / f"{name}_spikes.npz", **arrays)

        # Save parameters as JSON
        mp = self.model_params
        params_dict = {
            "realizations": self.realizations,
            "trials": self.trials,
            "model_params": {
                "N_E": mp.N_E,
                "N_I": mp.N_I,
                "duration_s": float(mp.duration),
                "analysis_start_t_s": float(mp.analysis_start_t),
                "analysis_window_t_s": float(mp.analysis_window_t),
                "fano_factor_window_t_s": float(mp.fano_factor_window_t),
                "firing_rate_window_t_s": float(mp.firing_rate_window_t),
                "p_conn_e": mp.p_conn_e,
                "p_conn_i": mp.p_conn_i,
            },
            "cluster_params": {
                "enabled": self.cluster_params.enabled,
                "R_ee": self.cluster_params.R_ee,
                "cluster_size": self.cluster_params.cluster_size,
                "weight_scaling_factor": self.cluster_params.weight_scaling_factor,
            },
        }
        with open(run_dir / "params.json", "w") as f:
            json.dump(params_dict, f, indent=2)

        print(f"Saved experiment data to {run_dir}/")
        return run_dir


class Experiment:
    """Runs simulations and collects spike data."""

    def __init__(
        self,
        realizations=12,
        trials=9,
        model_params=None,
        cluster_params=None,
        seed=42,
    ):
        if model_params is None:
            model_params = ModelParams()
        if cluster_params is None:
            cluster_params = ClusterParams(enabled=True, R_ee=2.5)

        self.realizations = realizations
        self.trials = trials
        self.model_params = model_params
        self.cluster_params = cluster_params
        self.seed = seed

        # Initialize networks
        self.uniform_networks = self._initialize_networks(
            ClusterParams(enabled=False, R_ee=1.0), "uniform"
        )
        self.clustered_networks = self._initialize_networks(cluster_params, "clustered")

    def _initialize_networks(self, cluster_params, name):
        networks = []
        for r in range(self.realizations):
            network = NeuronNetwork(
                self.model_params, cluster_params, seed=self.seed + 1000 * r
            )
            print(f"Built {name} network {r + 1}/{self.realizations}")
            networks.append(network)
        return networks

    def run(self) -> SpikeData:
        """Run all simulations and return spike data."""
        uniform_spikes = self._run_networks(self.uniform_networks, "uniform")
        clustered_spikes = self._run_networks(self.clustered_networks, "clustered")

        spike_data = SpikeData(
            uniform=uniform_spikes,
            clustered=clustered_spikes,
            model_params=self.model_params,
            cluster_params=self.cluster_params,
            realizations=self.realizations,
            trials=self.trials,
        )

        spike_data.save()
        return spike_data

    def _run_networks(self, networks, name):
        """Run networks and collect spike data."""
        print(f"Running {name} networks...")
        all_spikes = []

        for r, network in enumerate(networks):
            trial_spikes = []
            for t in range(self.trials):
                network.run()
                spike_times = np.array(network.spike_monitor_e.t)
                spike_ids = np.array(network.spike_monitor_e.i)
                trial_spikes.append((spike_times, spike_ids))
                print(
                    f"  Realization {r + 1}/{len(networks)}, "
                    f"Trial {t + 1}/{self.trials}"
                )
            all_spikes.append(trial_spikes)

        return all_spikes


def load_spike_data_from_disk(run_dir=None, base_dir="data"):
    """Load SpikeData from a saved experiment directory in data/.

    Args:
        run_dir: folder name (e.g., "experiment_run_2026-02-06_17-01-11") or full path.
                 If None, the latest experiment_run_* folder in base_dir is used.
        base_dir: parent directory that stores experiment_run_* folders.

    Returns:
        (spike_data, run_path)
    """
    base_path = Path(base_dir)

    if run_dir is None:
        runs = sorted([p for p in base_path.glob("experiment_run_*") if p.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No experiment_run_* folders found in {base_path}")
        run_path = runs[-1]
    else:
        run_path = Path(run_dir)
        if not run_path.exists():
            run_path = base_path / run_dir
        if not run_path.exists():
            raise FileNotFoundError(f"Could not find run directory: {run_dir}")

    params = json.loads((run_path / "params.json").read_text())
    model = params["model_params"]
    cluster = params["cluster_params"]

    model_params = ModelParams(
        N_E=model["N_E"],
        N_I=model["N_I"],
        duration=model["duration_s"] * second,
        analysis_start_t=model["analysis_start_t_s"] * second,
        analysis_window_t=model["analysis_window_t_s"] * second,
        fano_factor_window_t=model["fano_factor_window_t_s"] * second,
        firing_rate_window_t=model["firing_rate_window_t_s"] * second,
        p_conn_e=model["p_conn_e"],
        p_conn_i=model["p_conn_i"],
    )

    cluster_params = ClusterParams(
        enabled=cluster["enabled"],
        R_ee=cluster["R_ee"],
        cluster_size=cluster["cluster_size"],
        weight_scaling_factor=cluster["weight_scaling_factor"],
    )

    realizations = params["realizations"]
    trials = params["trials"]

    def _load_network_spikes(network_name):
        npz = np.load(run_path / f"{network_name}_spikes.npz")
        spikes = [[None for _ in range(trials)] for _ in range(realizations)]

        for r in range(realizations):
            for t in range(trials):
                times_key = f"r{r}_t{t}_times"
                ids_key = f"r{r}_t{t}_ids"
                if times_key not in npz or ids_key not in npz:
                    raise KeyError(
                        f"Missing keys {times_key}/{ids_key} "
                        f"in {network_name}_spikes.npz"
                    )
                spikes[r][t] = (
                    np.asarray(npz[times_key]),
                    np.asarray(npz[ids_key], dtype=np.int32),
                )

        return spikes

    spike_data = SpikeData(
        uniform=_load_network_spikes("uniform"),
        clustered=_load_network_spikes("clustered"),
        model_params=model_params,
        cluster_params=cluster_params,
        realizations=realizations,
        trials=trials,
    )

    print(f"Loaded spike_data from: {run_path}")
    print(
        f"  realizations={spike_data.realizations}, trials={spike_data.trials}, "
        f"N_E={spike_data.model_params.N_E}"
    )
    return spike_data, run_path
