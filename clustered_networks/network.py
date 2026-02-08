import numpy as np
from brian2 import (
    start_scope,
    NeuronGroup,
    Synapses,
    StateMonitor,
    SpikeMonitor,
    Network,
)

from .params import ModelParams, ClusterParams


class NeuronNetwork:
    def __init__(
        self, params: ModelParams, clustering_params: ClusterParams, seed=None
    ):
        self.params = params
        self.cluster_params = clustering_params
        self.rng = np.random.default_rng(seed=seed)
        self._initialize()

    def _initialize(self):
        start_scope()

        params = self.params
        cluster_params = self.cluster_params
        rng = self.rng

        # Unpack scaled weights
        j_ee, j_ei, j_ie, j_ii = params.scaled_weights

        # Set up equations
        eqs_base = """
        dV/dt = (mu - V) / tau_m + Isyn : 1 (unless refractory)
        Isyn = g : 1/second
        dg/dt = (x - g) / tau_2 : 1/second
        dx/dt = -x / tau_1 : 1/second

        mu : 1      # per-neuron bias (external input)
        tau_m : second  # per-neuron membrane time constant
        tau_1 : second # varies depending on inhibitory or excitatory neuron
        tau_2 : second # varies depending on inhibitory or excitatory neuron
        """

        # Excitatory population
        E = NeuronGroup(
            params.N_E,
            eqs_base,
            threshold=f"V > {params.Vt}",
            reset=f"V = {params.Vr}",
            refractory=params.refractory_t,
            method="euler",
            dt=params.dt,
        )
        # Inhibitory population
        I = NeuronGroup(  # noqa: E741
            params.N_I,
            eqs_base,
            threshold=f"V > {params.Vt}",
            reset=f"V = {params.Vr}",
            refractory=params.refractory_t,
            method="euler",
            dt=params.dt,
        )

        # Set up bias and initial voltages randomly
        E.mu = rng.uniform(*params.mu_e_range, size=params.N_E)
        E.V = rng.uniform(params.Vr, params.Vt, size=params.N_E)
        I.mu = rng.uniform(*params.mu_i_range, size=params.N_I)
        I.V = rng.uniform(params.Vr, params.Vt, size=params.N_I)

        # Set time constants
        E.tau_m = params.tau_m_e
        E.tau_1 = params.tau1
        E.tau_2 = params.tau2_e
        I.tau_m = params.tau_m_i
        I.tau_1 = params.tau1
        I.tau_2 = params.tau2_i

        # Set up synapses for connections involving inhibitory neurons
        S_ie = Synapses(E, I, "w: 1", on_pre="x_post += w/tau_1", dt=params.dt)
        # I -> E
        S_ei = Synapses(I, E, "w: 1", on_pre="x_post += w/tau_1", dt=params.dt)
        # I -> I
        S_ii = Synapses(I, I, "w: 1", on_pre="x_post += w/tau_1", dt=params.dt)

        # Connect inhibitory synapses (same probability for all)
        S_ei.connect(p=params.p_conn_i)
        S_ie.connect(p=params.p_conn_i)
        S_ii.connect(p=params.p_conn_i)

        S_ie.w = j_ie
        S_ei.w = j_ei
        S_ii.w = j_ii

        self.S_ee = None
        self.S_ee_in = None
        self.S_ee_out = None

        # E -> E connections: handle clustering if enabled
        # Uses a single Synapses object with cluster-aware connectivity
        if cluster_params.enabled:
            S_ee_in = Synapses(E, E, "w: 1", on_pre="x_post += w/tau_1", dt=params.dt)
            S_ee_out = Synapses(E, E, "w: 1", on_pre="x_post += w/tau_1", dt=params.dt)

            p_in, p_out = cluster_params.calculate_p_in_out(params.N_E, params.p_conn_e)

            # Connect with different probabilities based on cluster membership
            S_ee_in.connect(
                condition="i // cluster_size == j // cluster_size and i != j",
                p="p_in",
                namespace={
                    "p_in": p_in,
                    "p_out": p_out,
                    "cluster_size": cluster_params.cluster_size,
                },
            )

            S_ee_out.connect(
                condition="i // cluster_size != j // cluster_size",
                p="p_out",
                namespace={
                    "p_in": p_in,
                    "p_out": p_out,
                    "cluster_size": cluster_params.cluster_size,
                },
            )

            # Set weights: stronger for within-cluster, baseline for between-cluster
            S_ee_in.w = j_ee * cluster_params.weight_scaling_factor
            S_ee_out.w = j_ee

            self.S_ee_in = S_ee_in
            self.S_ee_out = S_ee_out
        else:
            S_ee = Synapses(E, E, "w: 1", on_pre="x_post += w/tau_1", dt=params.dt)

            # No clustering: uniform connectivity
            S_ee.connect(condition="i != j", p=params.p_conn_e)
            S_ee.w = j_ee

            self.S_ee = S_ee

        # --- Monitors ---
        state_monitor_e = StateMonitor(E, variables=["V", "x"], record=[0, 1, 2])
        state_monitor_i = StateMonitor(I, variables=["V", "x"], record=[0, 1, 2])
        spike_monitor_e = SpikeMonitor(E)
        spike_monitor_i = SpikeMonitor(I)

        self.E = E
        self.I = I
        self.S_ie = S_ie
        self.S_ei = S_ei
        self.S_ii = S_ii
        self.state_monitor_e = state_monitor_e
        self.state_monitor_i = state_monitor_i
        self.spike_monitor_e = spike_monitor_e
        self.spike_monitor_i = spike_monitor_i

        network_objects = [
            E,
            I,
            S_ie,
            S_ei,
            S_ii,
            state_monitor_e,
            state_monitor_i,
            spike_monitor_e,
            spike_monitor_i,
        ]
        if self.S_ee is not None:
            network_objects.append(self.S_ee)
        if self.S_ee_in is not None:
            network_objects.append(self.S_ee_in)
        if self.S_ee_out is not None:
            network_objects.append(self.S_ee_out)
        self.network = Network(*network_objects)

        self.network.store("initial")  # Persist the initial configuration

    def _randomize_initial_conditions(self):
        params = self.params
        rng = self.rng
        self.E.V = rng.uniform(params.Vr, params.Vt, size=params.N_E)
        self.I.V = rng.uniform(params.Vr, params.Vt, size=params.N_I)

    def run(self, randomize_initial=True):
        self.network.restore(
            "initial"
        )  # On every run call, restore initial configuration
        if randomize_initial:
            self._randomize_initial_conditions()

        if len(self.params.stimulus_multipliers) == 0:
            self.network.run(self.params.duration)
        else:
            start = self.params.stimulus_time[0]
            end = self.params.stimulus_time[1]
            self.network.run(start)
            self.E.mu *= self.params.stimulus_multipliers

            self.network.run(end - start)
            self.E.mu /= self.params.stimulus_multipliers

            self.network.run(self.params.duration - end)

    def restore(self):
        self.network.restore("initial")

    def spikes(self):
        return self.spike_monitor_e, self.spike_monitor_i


def firing_rate(spike_monitor, num_in_population, start_t, window_t):
    """Calculate firing rate for each neuron in a population.

    Args:
        spike_monitor: Brian2 SpikeMonitor object
        num_in_population: Number of neurons in the population
        start_t: Start time for analysis window
        window_t: Duration of analysis window

    Returns:
        Array of firing rates (spikes/second) for each neuron
    """
    mask = (spike_monitor.t >= start_t) & (spike_monitor.t <= start_t + window_t)
    spikes = spike_monitor.i[mask]
    spike_counts = np.bincount(spikes, minlength=num_in_population)
    return spike_counts / window_t
