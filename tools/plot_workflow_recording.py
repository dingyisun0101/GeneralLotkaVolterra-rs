"""Verify and plot the signal stream of a completed Workflow recording."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def load_signal(recording: Path):
    metadata = json.loads((recording / "metadata.json").read_text())
    if metadata.get("status", {}).get("state") != "complete":
        raise ValueError("recording is not complete")
    stream = next(item for item in metadata["streams"] if item["name"] == "signal")
    rows = []
    for chunk in stream["chunks"]:
        path = recording / stream["directory"] / chunk["file"]
        data = path.read_bytes()
        expected = chunk["checksum"].removeprefix("sha256:")
        if len(data) != chunk["bytes"] or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"integrity check failed for {path}")
        records = [json.loads(line) for line in data.splitlines()]
        if len(records) != chunk["records"]:
            raise ValueError(f"record count mismatch for {path}")
        for record in records:
            abundance = record["values"]["abundance"]["data"]
            rows.append((record["iteration"], record.get("physical_time"), abundance))
    if any(left[0] >= right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("signal iterations are not strictly increasing")
    return rows


def write_csv(path: Path, rows):
    species = len(rows[0][2])
    with path.open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["iteration", "physical_time", *[f"species_{i}" for i in range(species)]])
        for iteration, physical_time, abundance in rows:
            writer.writerow([iteration, physical_time, *abundance])


def write_plot(path: Path, rows):
    if path.exists():
        raise FileExistsError(path)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("plotting requires matplotlib; CSV export has no dependencies") from error
    times = [physical_time if physical_time is not None else iteration for iteration, physical_time, _ in rows]
    species = len(rows[0][2])
    for index in range(species):
        plt.plot(times, [abundance[index] for _, _, abundance in rows], label=f"species {index}")
    plt.xlabel("physical time")
    plt.ylabel("abundance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--plot", type=Path)
    arguments = parser.parse_args()
    rows = load_signal(arguments.recording)
    if not rows:
        raise ValueError("signal stream is empty")
    write_csv(arguments.csv, rows)
    if arguments.plot is not None:
        write_plot(arguments.plot, rows)


if __name__ == "__main__":
    main()
