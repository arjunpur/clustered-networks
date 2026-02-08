from pathlib import Path

import matplotlib.pyplot as plt


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
    output_dir="report/figures",
    dpi=300,
    format="pdf",
):
    """Save all publication figures to disk.

    Args:
        spike_data: SpikeData object
        R_ee_values: Array of R_ee values for Fano vs R_ee plot
        mean_fano_factors: Array of mean Fano factors for each R_ee
        output_dir: Directory to save figures
        dpi: Resolution
        format: Image format (e.g., 'pdf', 'png')

    Returns:
        List of saved file paths
    """
    from .distributions import plot_fano_factor, plot_firing_rate_distribution
    from .fano import plot_fano_vs_ree, plot_fano_vs_window
    from .correlation import plot_correlation_all_pairs, plot_correlation_same_cluster
    from .raster import plot_rasters_from_spike_data

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PUBLICATION_RCPARAMS)
    plt.rcParams.update({"figure.dpi": dpi, "savefig.dpi": dpi})

    def save_figure(name, plot_func, *args, **kwargs):
        """Helper to save a single figure."""
        print(f"Saving {name.replace('_', ' ')}...")
        fig, _ = plot_func(*args, **kwargs)
        filepath = output_path / f"{name}.{format}"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return filepath

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

    saved_files = [
        save_figure(name, func, *args, **kwargs) for name, func, args, kwargs in figures
    ]

    print(f"\nSaved {len(saved_files)} figures to '{output_dir}/':")
    for f in saved_files:
        print(f"  - {f.name}")

    return saved_files
