import numpy as np
from brian2 import ms, second, volt
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ClusterParams:
    enabled: bool = False
    R_ee: float = 1.0  # ratio of p_in / p_out. 1.0 implies no clustering
    cluster_size: int = 80  # num neurons in each cluster
    weight_scaling_factor: float = (
        1.9  # factor to scale excitatory synaptic weight for in-cluster connections
    )

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
        stimulus_time=(0, 0),
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
