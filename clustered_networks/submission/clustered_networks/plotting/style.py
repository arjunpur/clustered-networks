from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


PUBLICATION_RCPARAMS = {
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def save_all_figures(
    spike_data,
    R_ee_values,
    mean_fano_factors,
    stimulus_fano_data=None,
    covariance_kwargs=None,
    correlation_tail_kwargs=None,
    output_dir="report/figures",
    dpi=300,
    format="pdf",
):
    """Save all publication figures to disk.

    Args:
        spike_data: SpikeData object
        R_ee_values: Array of R_ee values for Fano vs R_ee plot
        mean_fano_factors: Array of mean Fano factors for each R_ee
        stimulus_fano_data: Optional dict with precomputed stimulus-response
            Fano curves. Required keys:
                timepoints_s
                clustered_mean
                clustered_lower
                clustered_upper
                uniform_mean
                uniform_lower
                uniform_upper
            Optional keys:
                title
                figsize
        covariance_kwargs: Optional dict overriding covariance computation args.
        correlation_tail_kwargs: Optional dict overriding tail diagnostics args.
        output_dir: Directory to save figures
        dpi: Resolution
        format: Image format (e.g., 'pdf', 'png')

    Returns:
        List of saved file paths
    """
    from ..analysis import compute_covariance_uniform_vs_clustered_from_spike_data
    from .correlation import (
        plot_correlation_all_pairs,
        plot_correlation_same_cluster,
        plot_correlation_tail,
    )
    from .covariance import (
        plot_autocovariance_comparison,
        plot_crosscovariance_comparison,
    )
    from .distributions import plot_fano_factor, plot_firing_rate_distribution
    from .fano import (
        plot_fano_stimulus_response,
        plot_fano_vs_ree,
        plot_fano_vs_window,
    )
    from .raster import plot_rasters_from_spike_data

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PUBLICATION_RCPARAMS)
    plt.rcParams.update({"figure.dpi": dpi, "savefig.dpi": dpi})

    covariance_defaults = {
        "bin_size_ms": 2,
        "max_lag_ms": 200,
        "max_pairs": 2000,
        "seed": 0,
        "normalize_by_mean_rate": False,
        "clustered_within_cluster_pairs": True,
    }
    covariance_config = {
        **covariance_defaults,
        **(covariance_kwargs or {}),
    }

    correlation_tail_defaults = {
        "subset_size": 500,
        "corr_window": 0.05,
        "corr_step": 0.025,
        "seed": 0,
        "title": "Correlation Tail Diagnostics: Same-Cluster vs Different-Cluster Pairs",
        "figsize": (12, 4.5),
        "show": False,
        "return_figure": True,
    }
    correlation_tail_config = {
        **correlation_tail_defaults,
        **(correlation_tail_kwargs or {}),
    }
    correlation_tail_config["show"] = False
    correlation_tail_config["return_figure"] = True

    covariance_curves_cache = None

    def _compute_covariance_curves(data):
        nonlocal covariance_curves_cache
        if covariance_curves_cache is not None:
            return covariance_curves_cache

        (
            lags_ms,
            uniform_auto,
            uniform_cross,
            clustered_auto,
            clustered_cross,
        ) = compute_covariance_uniform_vs_clustered_from_spike_data(
            data,
            bin_size_ms=covariance_config["bin_size_ms"],
            max_lag_ms=covariance_config["max_lag_ms"],
            max_pairs=covariance_config["max_pairs"],
            seed=covariance_config["seed"],
            normalize_by_mean_rate=covariance_config["normalize_by_mean_rate"],
            clustered_within_cluster_pairs=covariance_config[
                "clustered_within_cluster_pairs"
            ],
        )
        covariance_curves_cache = (
            lags_ms,
            uniform_auto,
            uniform_cross,
            clustered_auto,
            clustered_cross,
        )
        return covariance_curves_cache

    def _plot_autocovariance_from_spike_data(data):
        (
            lags_ms,
            uniform_auto,
            _uniform_cross,
            clustered_auto,
            _clustered_cross,
        ) = _compute_covariance_curves(data)

        fig, ax = plot_autocovariance_comparison(
            lags_ms,
            uniform_auto,
            clustered_auto,
            normalize_by_mean_rate=covariance_config["normalize_by_mean_rate"],
            title="Autocovariance: Uniform vs Clustered Networks",
        )
        fig.set_size_inches(6.5, 4.5)
        return fig, ax

    def _plot_crosscovariance_from_spike_data(data):
        (
            lags_ms,
            _uniform_auto,
            uniform_cross,
            _clustered_auto,
            clustered_cross,
        ) = _compute_covariance_curves(data)

        fig, ax = plot_crosscovariance_comparison(
            lags_ms,
            uniform_cross,
            clustered_cross,
            normalize_by_mean_rate=covariance_config["normalize_by_mean_rate"],
            title="Cross-Covariance: Uniform vs Clustered Networks",
        )
        fig.set_size_inches(6.5, 4.5)
        return fig, ax

    def _extract_figure(plot_result):
        if isinstance(plot_result, Figure):
            return plot_result
        if isinstance(plot_result, tuple) and plot_result:
            first = plot_result[0]
            if isinstance(first, Figure):
                return first

        fig = plt.gcf()
        if not isinstance(fig, Figure):
            raise TypeError("Plot function did not return a matplotlib Figure.")
        return fig

    def save_figure(name, plot_func, *args, **kwargs):
        """Helper to save a single figure."""
        print(f"Saving {name.replace('_', ' ')}...")
        fig = _extract_figure(plot_func(*args, **kwargs))
        filepath = output_path / f"{name}.{format}"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def _validate_stimulus_fano_data(data):
        required = [
            "timepoints_s",
            "clustered_mean",
            "clustered_lower",
            "clustered_upper",
            "uniform_mean",
            "uniform_lower",
            "uniform_upper",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            missing_keys = ", ".join(missing)
            raise ValueError(
                "stimulus_fano_data is missing required keys: "
                f"{missing_keys}."
            )

    figures = [
        ("firing_rate_distribution", plot_firing_rate_distribution, (spike_data,), {}),
        ("fano_factor_distribution", plot_fano_factor, (spike_data,), {}),
        ("correlation_all_pairs", plot_correlation_all_pairs, (spike_data,), {}),
        (
            "correlation_same_cluster",
            plot_correlation_same_cluster,
            (spike_data,),
            {},
        ),
        ("fano_factor_vs_window", plot_fano_vs_window, (spike_data,), {}),
        (
            "fano_factor_vs_ree",
            plot_fano_vs_ree,
            (R_ee_values, mean_fano_factors),
            {},
        ),
        (
            "autocovariance_comparison",
            _plot_autocovariance_from_spike_data,
            (spike_data,),
            {},
        ),
        (
            "crosscovariance_comparison",
            _plot_crosscovariance_from_spike_data,
            (spike_data,),
            {},
        ),
        (
            "correlation_tail_diagnostics",
            plot_correlation_tail,
            (spike_data,),
            correlation_tail_config,
        ),
        (
            "rasters_uniform",
            plot_rasters_from_spike_data,
            (spike_data,),
            {"network_type": "uniform"},
        ),
        (
            "rasters_clustered",
            plot_rasters_from_spike_data,
            (spike_data,),
            {"network_type": "clustered"},
        ),
    ]

    if stimulus_fano_data is not None:
        _validate_stimulus_fano_data(stimulus_fano_data)
        figures.append(
            (
                "fano_factor_higher_input_stimulus_response",
                plot_fano_stimulus_response,
                (),
                {
                    "timepoints_s": stimulus_fano_data["timepoints_s"],
                    "clustered_mean": stimulus_fano_data["clustered_mean"],
                    "clustered_lower": stimulus_fano_data["clustered_lower"],
                    "clustered_upper": stimulus_fano_data["clustered_upper"],
                    "uniform_mean": stimulus_fano_data["uniform_mean"],
                    "uniform_lower": stimulus_fano_data["uniform_lower"],
                    "uniform_upper": stimulus_fano_data["uniform_upper"],
                    "title": stimulus_fano_data.get(
                        "title",
                        "Fano Factor Reaction to Higher Input Stimulus",
                    ),
                    "figsize": stimulus_fano_data.get("figsize", (8, 4.5)),
                },
            )
        )
    else:
        print(
            "Skipping fano factor higher input stimulus response "
            "(stimulus_fano_data not provided)."
        )

    saved_files = [
        save_figure(name, func, *args, **kwargs) for name, func, args, kwargs in figures
    ]

    print(f"\nSaved {len(saved_files)} figures to '{output_dir}/':")
    for f in saved_files:
        print(f"  - {f.name}")

    return saved_files
