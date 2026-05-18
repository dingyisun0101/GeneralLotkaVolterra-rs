"""Render plots from signal JSON outputs produced by the Rust examples.

This script accepts an output directory, a signal directory, or a single signal
JSON file. When given an output directory with a `signal/` child, it loads every
numeric signal JSON in order, concatenates the full simulation in memory, and
calls the plotter to produce `plot.png`.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from typing import Iterable, Tuple
try:
    from .plotter import plot_frequency_and_heatmap
except ImportError:
    from plotter import plot_frequency_and_heatmap


def load_json_samples(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with json_path.open() as f:
        j = json.load(f)

    samples = j.get("samples")
    if not samples:
        raise ValueError("No 'samples' key found in signal JSON")

    times = []
    states = []
    for s in samples:
        times.append(float(s.get("time", 0)))
        st = s.get("state")
        if st and "data" in st:
            data = st["data"]
            states.append(np.asarray(data, dtype=float).ravel())
            continue

        raise ValueError("Could not find state data in signal sample entry")

    t = np.array(times)
    nu = np.vstack(states)
    # Convert absolute abundances to frequencies per time (simple normalization)
    row_sums = nu.sum(axis=1, keepdims=True)
    # avoid division by zero
    row_sums[row_sums == 0] = 1.0
    freqs = nu / row_sums
    return t, freqs


def load_json_series(json_paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    times = []
    states = []
    previous_end = None

    for json_path in json_paths:
        t, nu = load_json_samples(json_path)
        if previous_end is not None and t.size > 0 and t[0] <= previous_end:
            keep = t > previous_end
            t = t[keep]
            nu = nu[keep, :]

        if t.size == 0:
            continue

        times.append(t)
        states.append(nu)
        previous_end = float(t[-1])

    if not times:
        raise ValueError("No samples found in JSON series")

    return np.concatenate(times), np.vstack(states)


def epoch_number(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def render_from_path(path: str, *, out_plot_dir: str = None, title: str = None):
    p = Path(path)
    if p.is_dir():
        data_dir = p
        signal_dir = p / "signal"
        if signal_dir.is_dir():
            p = signal_dir

        json_files = sorted(
            (json_file for json_file in p.glob("*.json") if epoch_number(json_file) >= 0),
            key=epoch_number,
        )
        if not json_files:
            raise SystemExit(f"No JSON files found in {p}")
        t, nu = load_json_series(json_files)
        default_title = data_dir.name
    else:
        t, nu = load_json_samples(p)
        data_dir = p.parent
        default_title = p.stem

    # determine outdir for plots
    if out_plot_dir:
        outdir = Path(out_plot_dir)
    else:
        outdir = data_dir / "plot"

    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_frequency_and_heatmap(t, nu, outdir, title=title or default_title)
    return plot_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Render example plots from JSON output")
    p.add_argument("path", help="path to output directory or JSON file")
    p.add_argument("--outdir", help="optional output plot directory")
    p.add_argument("--title", help="optional plot title")
    args = p.parse_args()

    out = render_from_path(args.path, out_plot_dir=args.outdir, title=args.title)
    print(out)
