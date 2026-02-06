import numpy as np
from brian2 import *
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ClusterParams:
    enabled: bool = False
    R_ee: float = 1.0  # ratio of p_in / p_out. 1.0 implies no clustering
    cluster_size: int = 80  # num neurons in each cluster
    weight_scaling_factor: float = 1.9  # factor to scale excitatory synaptic weight for in-cluster connections

    def calculate_p_in_out(self, N_E: int, p_avg: float) -> Tuple[float, float]:
        # Use within-cluster pair probability (exclude self)
        p_same = (self.cluster_size - 1) / (N_E - 1)

        # Enforce: p_avg = p_same * p_in + (1 - p_same) * p_out
        # and p_in = R_ee * p_out
        denom = p_same * self.R_ee + (1 - p_same)
        p_out = p_avg / denom
        p_in = self.R_ee * p_out

        return p_in, p_out


class ModelParams:
    """Parameters for the clustered network model."""

    def __init__(
        self,
        # Network size
        N_E=4000,  # number of excitatory neurons
        N_I=1000,  # number of inhibitory neurons
        N_ref=5000,  # reference network size (for weight scaling)
        # Neuron parameters
        refractory_t=5 * ms,  # refractory period
        tau_e=15 * ms,  # membrane time constant for E neurons
        tau_i=10 * ms,  # membrane time constant for I neurons
        Vt=1,  # spike threshold
        Vr=0,  # reset potential
        # Synaptic time constants
        tau_1=1 * ms,  # exponential filter time constant 1
        tau2_e=3 * ms,  # exponential filter time constant 2 (excitatory)
        tau2_i=2 * ms,  # exponential filter time constant 2 (inhibitory)
        # Reference synaptic weights (calibrated for N=5000)
        j_ee_ref=0.024,  # E -> E
        j_ie_ref=0.014,  # E -> I
        j_ei_ref=-0.045,  # I -> E (negative for inhibition)
        j_ii_ref=-0.057,  # I -> I
        # Connection probabilities
        p_conn_e=0.2,  # E -> E connection probability
        p_conn_i=0.5,  # connection probability involving I
        # Mean equilibrium
        mu_e_range=(1.1, 1.2),  # external input range for E neurons
        mu_i_range=(1.0, 1.05),  # external input range for I neurons
        # Simulation parameters
        dt=0.1 * ms,  # integration timestep
        analysis_start_t=1.5 * second,  # time to start statistical analysis
        analysis_window_t=1.5 * second,  # time window for correlation analysis
        fano_factor_window_t=100 * ms,
        firing_rate_window_t=1.5 * second,  # time window for firing rate analysis
        duration=3 * second,  # simulation duration
        # Plotting
        voltage_scale=15 * volt,  # for converting to physical units
        stimulus_multipliers=np.array([]),
        stimulus_time=(0,0),
    ):
        self.N_E = N_E
        self.N_I = N_I
        self.N_ref = N_ref
        self.refractory_t = refractory_t
        self.tau_m_e = tau_e
        self.tau_m_i = tau_i
        self.Vt = Vt
        self.Vr = Vr
        self.tau1 = tau_1
        self.tau2_e = tau2_e
        self.tau2_i = tau2_i
        self.j_ee_ref = j_ee_ref
        self.j_ei_ref = j_ei_ref
        self.j_ie_ref = j_ie_ref
        self.j_ii_ref = j_ii_ref
        self.p_conn_e = p_conn_e
        self.p_conn_i = p_conn_i
        self.mu_e_range = mu_e_range
        self.mu_i_range = mu_i_range
        self.dt = dt
        self.analysis_start_t = analysis_start_t
        self.analysis_window_t = analysis_window_t
        self.fano_factor_window_t = fano_factor_window_t
        self.firing_rate_window_t = firing_rate_window_t
        self.duration = duration
        self.voltage_scale = voltage_scale
        self.stimulus_multipliers = stimulus_multipliers
        self.stimulus_time = stimulus_time

    @property
    def N_total(self):
        """Total number of neurons."""
        return self.N_E + self.N_I

    @property
    def scaled_weights(self):
        """
        Scale synaptic weights based on network size.

        Weights scale as 1/sqrt(N) to maintain constant variance
        of total synaptic input as network size changes.
        """
        scale_factor = np.sqrt(self.N_ref / self.N_total)
        return (
            self.j_ee_ref * scale_factor,
            self.j_ei_ref * scale_factor,
            self.j_ie_ref * scale_factor,
            self.j_ii_ref * scale_factor,
        )


class NeuronNetwork:
    def __init__(self, params: ModelParams, clustering_params: ClusterParams, seed=None):
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
        tau1 = params.tau1

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
        I = NeuronGroup(
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
        self.network.restore("initial")  # On every run call, restore initial configuration
        if randomize_initial:
            self._randomize_initial_conditions()

        if len(self.params.stimulus_multipliers) == 0:
            self.network.run(self.params.duration)
        else:
            start = self.params.stimulus_time[0]
            end = self.params.stimulus_time[1]
            self.network.run(start)
            self.E.mu *= self.params.stimulus_multipliers

            self.network.run(end-start)
            self.E.mu /= self.params.stimulus_multipliers

            self.network.run(self.params.duration-end)

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
