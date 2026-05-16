"""Lightweight plotter for frequency evolution and sampled state heatmap.

Usage from other Python code:
    from examples.common.plotter import plot_frequency_and_heatmap
    plot_frequency_and_heatmap(t, nu, outdir="output/example/plot", title="My Run")

Or CLI:
    python -m examples.common.plotter --input data.npz --outdir output/example/plot

The script expects `t` (1D) and `nu` (2D: T x K) arrays. If `nu` is K x T it will be transposed.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _bin_time_series(t, nu, bins):
    """Average every row into chronological bins so every sample contributes."""
    bins = max(1, min(int(bins), nu.shape[0]))
    row_bins = np.array_split(np.arange(nu.shape[0]), bins)
    binned_t = np.array([t[rows].mean() for rows in row_bins])
    binned_nu = np.vstack([nu[rows, :].mean(axis=0) for rows in row_bins])
    return binned_t, binned_nu


def _autoscale_frequency_axis(ax, nu, *, logscale, eps):
    positive = nu[nu > eps]
    if positive.size == 0:
        ax.set_ylim(0.0, 1.0)
        return False

    y_max = min(1.0, max(float(positive.max()) * 1.08, eps * 10.0))
    y_floor = max(eps, float(np.percentile(positive, 1)) * 0.5)
    dynamic_range = y_max / y_floor if y_floor > 0 else np.inf

    if logscale and dynamic_range >= 100.0:
        ax.set_yscale("log")
        ax.set_ylim(y_floor, y_max)
        return True

    ax.set_yscale("linear")
    y_min = max(0.0, float(np.percentile(nu, 1)) * 0.95)
    y_max = min(1.0, max(float(np.percentile(nu, 99)) * 1.08, float(nu.max()) * 1.02))
    if y_max <= y_min:
        y_max = y_min + 1.0
    ax.set_ylim(y_min, y_max)
    return False


def plot_frequency_and_heatmap(t, nu, outdir, *, title=None, sample_n=20, logscale=True, eps=1e-12, figsize=(10, 6)):
    """Save a two-panel figure showing frequency time-series and sampled state heatmap.

    Args:
        t: 1D time array of length T.
        nu: 2D array with shape (T, K) or (K, T). Rows = times, cols = strains.
        outdir: directory where `plot.png` will be saved (created if missing).
        title: optional figure title.
        sample_n: number of chronological bins to show in the heatmap.
        logscale: whether to plot the time-series with a log y-axis.
        eps: small value added before log transforms to avoid -inf.
        figsize: matplotlib figure size tuple.
    Returns:
        Path to saved plot file.
    """
    t = np.asarray(t)
    nu = np.asarray(nu)
    # Normalize input shapes: want (T, K)
    if nu.ndim != 2:
        raise ValueError("`nu` must be 2D (T x K)")
    if nu.shape[0] == t.size and nu.shape[1] != t.size:
        T, K = nu.shape
    elif nu.shape[1] == t.size and nu.shape[0] != t.size:
        nu = nu.T
        T, K = nu.shape
    else:
        # ambiguous; assume rows are times
        T, K = nu.shape

    if t.size != T:
        raise ValueError("`t` length must match one axis of `nu`")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.38})

    ax_ts = axes[0]
    # Plot each strain frequency over time
    for k in range(K):
        ax_ts.plot(t, nu[:, k], lw=0.9, alpha=0.9)
    ax_ts.set_ylabel("frequency")
    ax_ts.set_xlabel("time")
    if t.size > 0:
        ax_ts.set_xlim(float(t[0]), float(t[-1]))
    use_log = _autoscale_frequency_axis(ax_ts, nu, logscale=logscale, eps=eps)
    if use_log:
        for line in ax_ts.lines:
            line.set_ydata(np.maximum(line.get_ydata(), ax_ts.get_ylim()[0]))
    ax_ts.grid(True, which="both", ls=":", lw=0.5)

    if title:
        ax_ts.set_title(title)

    # Heatmap: bin all times into chronological bins and draw earliest at bottom.
    ax_hm = axes[1]
    _, binned_nu = _bin_time_series(t, nu, sample_n)
    bin_count = binned_nu.shape[0]
    time_min = float(t[0])
    time_max = float(t[-1])
    if time_max <= time_min:
        time_max = time_min + 1.0

    # Order columns (strains) by final abundance descending to match notebook style
    order = np.argsort(-nu[-1, :])
    sampled_ordered = binned_nu[:, order]

    # Show heatmap of log10 frequencies for dynamic range visibility
    log_sampled = np.log10(sampled_ordered + eps)

    im = ax_hm.imshow(
        log_sampled,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="lower",
        extent=(-0.5, K - 0.5, time_min, time_max),
    )
    ax_hm.set_ylabel("time")
    ax_hm.set_xlabel("strain (ordered by final abundance)")
    cbar = fig.colorbar(im, ax=ax_hm, orientation="vertical", pad=0.02)
    cbar.set_label("log10(frequency)")

    # X ticks: show a few strain indices
    xticks = np.linspace(0, K - 1, min(8, K), dtype=int)
    ax_hm.set_xticks(xticks)
    ax_hm.set_xticklabels([str(i) for i in order[xticks]])

    # Y ticks: time increases upward because origin="lower".
    yticks = np.linspace(time_min, time_max, min(6, bin_count))
    ax_hm.set_yticks(yticks)
    ax_hm.set_yticklabels([f"{value:.2f}" for value in yticks])

    outpath = outdir / "plot.png"
    fig.subplots_adjust(hspace=0.38)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _cli():
    p = argparse.ArgumentParser(description="Plot frequency time-series and sampled-state heatmap")
    p.add_argument("--input", required=True, help=".npz file containing arrays 't' and 'nu' (T x K)")
    p.add_argument("--outdir", required=True, help="output directory to save plot")
    p.add_argument("--title", default=None)
    p.add_argument("--samples", type=int, default=20)
    args = p.parse_args()

    data = np.load(args.input)
    if "t" not in data or "nu" not in data:
        raise SystemExit("Input .npz must contain arrays named 't' and 'nu'")

    t = data["t"]
    nu = data["nu"]
    outpath = plot_frequency_and_heatmap(t, nu, args.outdir, title=args.title, sample_n=args.samples)
    print(str(outpath))


if __name__ == "__main__":
    _cli()
