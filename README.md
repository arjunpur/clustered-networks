# clustered-networks

Neural network simulations studying slow dynamics and high variability in
networks with clustered connections, using [Brian2](https://brian2.readthedocs.io/).

## Project Structure

```
clustered_networks/           # Python package
    params.py                 # ClusterParams, ModelParams
    network.py                # NeuronNetwork, firing_rate (Brian2)
    experiment.py             # SpikeData, Experiment, load/save I/O
    analysis/                 # Pure-numpy analysis (no Brian2, no matplotlib)
        firing_rate.py        # compute_firing_rates
        fano.py               # compute_fano_factor, compute_fano_factor_for_ree
        correlation.py        # Correlation coefficient functions
        covariance.py         # Binning, auto/cross-covariance, pair sampling
        tail.py               # analyze_correlation_tail
    plotting/                 # Matplotlib visualizations
        raster.py             # Spike raster plots
        distributions.py      # Firing rate & Fano factor histograms
        fano.py               # Fano factor vs window / R_ee
        correlation.py        # Correlation coefficient histograms & tail
        covariance.py         # Autocovariance & cross-covariance plots
        style.py              # Publication rcParams, save_all_figures
base_model.ipynb              # Narrative notebook (imports from package)
```

## Prerequisites

- Python 3.13 or higher

## Setup

### 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

This will create a virtual environment and install all required packages.

### 3. Run the notebook

```bash
make run
# or: uv run jupyter lab base_model.ipynb
```

## Reproducing Results

1. **Run the experiment** by executing the notebook cells in order. This will
   save spike data under `data/experiment_run_<timestamp>/`.

2. **Load saved data** instead of re-running:
   ```python
   from clustered_networks import load_spike_data_from_disk
   spike_data, run_path = load_spike_data_from_disk()
   ```

## Development

```bash
make lint      # ruff check
make format    # ruff format
```
