import numpy as np
import matplotlib.pyplot as plt


def _validate_covariance_curve_shapes(lags_ms, *curves):
    lags_ms = np.asarray(lags_ms)
    if lags_ms.ndim != 1:
        raise ValueError(f"lags_ms must be 1D, got shape {lags_ms.shape}.")

    for name, curve in curves:
        arr = np.asarray(curve)
        if arr.shape != lags_ms.shape:
            raise ValueError(
                f"{name} must have shape {lags_ms.shape}, got {arr.shape}."
            )


def plot_covariance_comparison(
    lags_ms,
    uniform_auto,
    clustered_auto,
    uniform_cross,
    clustered_cross,
    normalize_by_mean_rate=False,
    axes=None,
    title=None,
):
    """
    Plot autocovariance and cross-covariance comparisons.

    Cross-covariance convention in this plot:
      - uniform_cross: average cross-covariance in the uniform network (all sampled pairs)
      - clustered_cross: cross-covariance for same-cluster pairs in the clustered network
    """
    _validate_covariance_curve_shapes(
        lags_ms,
        ("uniform_auto", uniform_auto),
        ("clustered_auto", clustered_auto),
        ("uniform_cross", uniform_cross),
        ("clustered_cross", clustered_cross),
    )

    lags_ms = np.asarray(lags_ms)
    uniform_auto = np.asarray(uniform_auto)
    clustered_auto = np.asarray(clustered_auto)
    uniform_cross = np.asarray(uniform_cross)
    clustered_cross = np.asarray(clustered_cross)

    created_fig = False
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
        created_fig = True
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain exactly 2 Axes objects.")
        fig = axes[0].figure

    auto_ax, cross_ax = axes

    # Autocovariance panel
    auto_ax.plot(lags_ms, uniform_auto, color="black", linewidth=2, label="Uniform")
    auto_ax.plot(
        lags_ms, clustered_auto, color="limegreen", linewidth=2, label="Clustered"
    )
    auto_ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    auto_ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    auto_ax.set_xlabel("Lag (ms)")
    auto_ax.set_ylabel(
        "Covariance / mean rate" if normalize_by_mean_rate else "Covariance (Hz$^2$)"
    )
    auto_ax.set_title("Autocovariance")
    auto_ax.legend()

    # Cross-covariance panel
    cross_ax.plot(
        lags_ms,
        uniform_cross,
        color="black",
        linewidth=2,
        label="Uniform (average cross-covariance)",
    )
    cross_ax.plot(
        lags_ms,
        clustered_cross,
        color="dodgerblue",
        linewidth=2,
        label="Clustered (same-cluster pairs)",
    )
    cross_ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    cross_ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    cross_ax.set_xlabel("Lag (ms)")
    cross_ax.set_title("Cross-covariance")
    cross_ax.legend()

    if title is None:
        title = "Covariance Comparison: Uniform vs Clustered"
    fig.suptitle(title, y=1.03)

    if created_fig:
        plt.tight_layout()

    return fig, axes
