# clustered-networks

Neural network simulations using [Brian2](https://brian2.readthedocs.io/).

## Prerequisites

- Python 3.13 or higher

## Setup

### 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install dependencies

```bash
uv sync
```

This will create a virtual environment and install all required packages.

Then open any `.ipynb` file (e.g., `base_model.ipynb`).
