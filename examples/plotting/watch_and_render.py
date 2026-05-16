"""Watch a directory for example JSON outputs and render plots automatically.

This script polls the target directory (recursively) for new or modified `.json` files
and calls `render_from_output.render_from_path` to produce `plot/plot.png`.

Usage:
    python -m examples.plotting.watch_and_render GLV/output
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict
from .render_from_output import render_from_path


def scan_json_files(root: Path) -> Dict[Path, float]:
    out: Dict[Path, float] = {}
    for p in root.rglob("*.json"):
        try:
            out[p] = p.stat().st_mtime
        except OSError:
            continue
    return out


def watch_and_render(root: str, poll_interval: float = 5.0):
    rootp = Path(root)
    if not rootp.exists():
        raise SystemExit(f"Path does not exist: {root}")

    seen = scan_json_files(rootp)
    print(f"Watching {rootp} for new/changed .json files. Poll interval {poll_interval}s")
    try:
        while True:
            time.sleep(poll_interval)
            current = scan_json_files(rootp)
            # detect new or updated files
            for p, mtime in current.items():
                if p not in seen or seen[p] < mtime:
                    print(f"Detected new/changed JSON: {p}. Rendering...")
                    try:
                        render_from_path(str(p))
                        print(f"Rendered plot for {p}")
                    except Exception as e:
                        print(f"Error rendering {p}: {e}")
            seen = current
    except KeyboardInterrupt:
        print("Watcher stopped by user")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Watch GLV output directory and auto-render plots")
    p.add_argument("path", help="root directory to watch (e.g. GLV/output)")
    p.add_argument("--interval", type=float, default=5.0, help="poll interval seconds")
    args = p.parse_args()
    watch_and_render(args.path, poll_interval=args.interval)
